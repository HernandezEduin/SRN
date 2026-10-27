"""Train TransE on a KG and assess filtered link prediction before SRN export.

The SRN input remains an NPZ with aligned 'relations' and 'vectors' arrays.
No questions, QA splits, or annotated paths enter this pretraining step.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_triples(path):
    entities, relations, triples = {}, {}, []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 3:
                raise ValueError("Invalid triple on line {}".format(lineno))
            head, relation, tail = parts
            for entity in (head, tail):
                if entity not in entities:
                    entities[entity] = len(entities)
            if relation not in relations:
                relations[relation] = len(relations)
            triples.append((entities[head], relations[relation], entities[tail]))
    if not triples:
        raise ValueError("Graph contains no triples")
    # Duplicate triples cannot be placed in different held-out partitions.
    triples = np.unique(np.asarray(triples, dtype=np.int64), axis=0)
    return triples, entities, relations


def pretrain(triples, n_entities, n_relations, dim=50, epochs=100, batch_size=2048,
             learning_rate=0.01, margin=1.0, seed=42):
    """Legacy NumPy pretraining helper kept for backward compatibility.\n\n    The improved CLI uses train_transe(), not this historical helper.\n    """
    rng = np.random.default_rng(seed)
    entities = rng.uniform(-6 / np.sqrt(dim), 6 / np.sqrt(dim), (n_entities, dim)).astype(np.float32)
    relations = rng.uniform(-6 / np.sqrt(dim), 6 / np.sqrt(dim), (n_relations, dim)).astype(np.float32)
    entities /= np.maximum(np.linalg.norm(entities, axis=1, keepdims=True), 1e-12)
    for epoch in range(epochs):
        ordering = rng.permutation(len(triples))
        losses = []
        for offset in range(0, len(ordering), batch_size):
            positives = triples[ordering[offset:offset + batch_size]]
            negatives = positives.copy()
            corrupt_head = rng.random(len(positives)) < 0.5
            replacements = rng.integers(n_entities, size=len(positives), dtype=np.int32)
            negatives[corrupt_head, 0] = replacements[corrupt_head]
            negatives[~corrupt_head, 2] = replacements[~corrupt_head]
            positive_diff = entities[positives[:, 0]] + relations[positives[:, 1]] - entities[positives[:, 2]]
            negative_diff = entities[negatives[:, 0]] + relations[negatives[:, 1]] - entities[negatives[:, 2]]
            hinge = margin + np.abs(positive_diff).sum(axis=1) - np.abs(negative_diff).sum(axis=1)
            active = hinge > 0
            if not active.any():
                continue
            losses.append(float(hinge[active].mean()))
            pos, neg = positives[active], negatives[active]
            pg = np.sign(positive_diff[active]) * learning_rate
            ng = np.sign(negative_diff[active]) * learning_rate
            np.add.at(entities, pos[:, 0], -pg)
            np.add.at(relations, pos[:, 1], -pg)
            np.add.at(entities, pos[:, 2], pg)
            np.add.at(entities, neg[:, 0], ng)
            np.add.at(relations, neg[:, 1], ng)
            np.add.at(entities, neg[:, 2], -ng)
        entities /= np.maximum(np.linalg.norm(entities, axis=1, keepdims=True), 1e-12)
        print("TransE epoch {}/{} margin loss {:.4f}".format(epoch + 1, epochs, np.mean(losses) if losses else 0), flush=True)
    return relations



def split_triples(triples, n_entities, n_relations, valid_ratio, test_ratio, seed):
    """Hold out links while keeping every entity and relation in training."""
    if valid_ratio <= 0 or test_ratio <= 0 or valid_ratio + test_ratio >= 1:
        raise ValueError("Require positive valid/test ratios whose sum is below 1")
    count = len(triples)
    targets = (max(1, round(count * valid_ratio)), max(1, round(count * test_ratio)))
    if sum(targets) >= count:
        raise ValueError("Not enough triples for the requested held-out split")
    rng = np.random.default_rng(seed)
    degrees = np.bincount(
        np.concatenate((triples[:, 0], triples[:, 2])), minlength=n_entities
    )
    relation_counts = np.bincount(triples[:, 1], minlength=n_relations)
    heldout = [[], []]
    train_mask = np.ones(count, dtype=bool)
    for index in rng.permutation(count):
        partition = 0 if len(heldout[0]) < targets[0] else 1
        if len(heldout[partition]) >= targets[partition]:
            break
        head, relation, tail = map(int, triples[index])
        head_usage = 1 + int(head == tail)
        if degrees[head] <= head_usage or relation_counts[relation] <= 1:
            continue
        if head != tail and degrees[tail] <= 1:
            continue
        heldout[partition].append(index)
        train_mask[index] = False
        degrees[head] -= head_usage
        if head != tail:
            degrees[tail] -= 1
        relation_counts[relation] -= 1
    if any(len(indices) < target for indices, target in zip(heldout, targets)):
        raise ValueError("Could not hold out enough triples while preserving entity/relation coverage")
    return triples[train_mask], triples[heldout[0]], triples[heldout[1]]


def triple_keys(triples, n_entities, n_relations):
    triples = np.asarray(triples, dtype=np.int64)
    return ((triples[..., 0] * n_relations + triples[..., 1]) *
            n_entities + triples[..., 2])


def sample_negatives(positives, n_entities, n_relations, negative_size, known_keys, rng):
    """Uniform head/tail corruption excluding known training triples."""
    shape = (len(positives), negative_size)
    negatives = np.broadcast_to(positives[:, None, :], shape + (3,)).copy()
    corrupt_head = rng.random(shape) < 0.5
    replacements = rng.integers(n_entities, size=shape)
    negatives[..., 0] = np.where(corrupt_head, replacements, negatives[..., 0])
    negatives[..., 2] = np.where(~corrupt_head, replacements, negatives[..., 2])
    for _ in range(100):
        keys = triple_keys(negatives, n_entities, n_relations)
        positions = np.searchsorted(known_keys, keys)
        is_positive = ((positions < len(known_keys)) &
                       (known_keys[np.minimum(positions, len(known_keys) - 1)] == keys))
        if not is_positive.any():
            return negatives
        replacements = rng.integers(n_entities, size=int(is_positive.sum()))
        # Keep the chosen corruption side consistent for each negative sample.
        head_mask = is_positive & corrupt_head
        tail_mask = is_positive & ~corrupt_head
        negatives[..., 0][head_mask] = replacements[:int(head_mask.sum())]
        negatives[..., 2][tail_mask] = replacements[int(head_mask.sum()):]
    raise RuntimeError("Unable to sample non-positive corruptions. Check the graph density.")


def filtered_link_prediction(entities, relations, test_triples, all_triples,
                             max_examples=256, seed=42):
    """Filtered entity ranking for both (h,r,?) and (?,r,t), no sampling of candidates."""
    rng = np.random.default_rng(seed)
    evaluated = test_triples
    if max_examples > 0 and len(test_triples) > max_examples:
        indices = np.sort(rng.choice(len(test_triples), size=max_examples, replace=False))
        evaluated = test_triples[indices]
    known_heads, known_tails = defaultdict(set), defaultdict(set)
    for head, rel, tail in all_triples:
        known_heads[(int(rel), int(tail))].add(int(head))
        known_tails[(int(head), int(rel))].add(int(tail))
    ranks = []
    for head, rel, tail in evaluated:
        head, rel, tail = int(head), int(rel), int(tail)
        target_tail = entities[head] + relations[rel]
        tail_dist = np.abs(entities - target_tail).sum(axis=1)
        true_tail_dist = tail_dist[tail]
        tail_dist[list(known_tails[(head, rel)])] = np.inf
        tail_dist[tail] = true_tail_dist
        tail_rank = (1 + np.count_nonzero(tail_dist < true_tail_dist) +
                     0.5 * (np.count_nonzero(tail_dist == true_tail_dist) - 1))
        target_head = entities[tail] - relations[rel]
        head_dist = np.abs(entities - target_head).sum(axis=1)
        true_head_dist = head_dist[head]
        head_dist[list(known_heads[(rel, tail)])] = np.inf
        head_dist[head] = true_head_dist
        head_rank = (1 + np.count_nonzero(head_dist < true_head_dist) +
                     0.5 * (np.count_nonzero(head_dist == true_head_dist) - 1))
        ranks.extend((head_rank, tail_rank))
    ranks = np.asarray(ranks, dtype=np.float64)
    if not len(ranks):
        raise ValueError("Link-prediction split contains no triples")
    return {
        "triples_evaluated": int(len(evaluated)),
        "predictions_evaluated": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mr": float(np.mean(ranks)),
        "hits_at_1": float(np.mean(ranks <= 1)),
        "hits_at_3": float(np.mean(ranks <= 3)),
        "hits_at_10": float(np.mean(ranks <= 10)),
    }


def train_transe(triples, n_entities, n_relations, args, epochs, seed,
                 valid_triples=None, filter_triples=None):
    """TensorFlow TransE with head/tail negatives and logistic link-prediction loss."""
    import tensorflow as tf
    from tqdm.auto import tqdm

    tf.random.set_seed(seed)
    print("TransE device availability: GPUs {}".format(
        [device.name for device in tf.config.list_physical_devices("GPU")]), flush=True)
    rng = np.random.default_rng(seed)
    radius = 6.0 / np.sqrt(args.dim)
    entity_init = rng.uniform(-radius, radius, (n_entities, args.dim)).astype(np.float32)
    entity_init /= np.maximum(np.linalg.norm(entity_init, axis=1, keepdims=True), 1e-12)
    relation_init = rng.uniform(-radius, radius, (n_relations, args.dim)).astype(np.float32)
    entity_var = tf.Variable(entity_init, name="entities")
    relation_var = tf.Variable(relation_init, name="relations")
    print("TransE entity variable device: {}".format(entity_var.device), flush=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    known_keys = np.unique(triple_keys(triples, n_entities, n_relations))
    logs, best = [], None
    patience_count = 0

    @tf.function(reduce_retracing=True)
    def step(pos, neg):
        with tf.GradientTape() as tape:
            ph = tf.gather(entity_var, pos[:, 0])
            pr = tf.gather(relation_var, pos[:, 1])
            pt = tf.gather(entity_var, pos[:, 2])
            positive_logits = args.gamma - tf.reduce_sum(tf.abs(ph + pr - pt), axis=-1)
            nh = tf.gather(entity_var, neg[:, :, 0])
            nr = tf.gather(relation_var, neg[:, :, 1])
            nt = tf.gather(entity_var, neg[:, :, 2])
            negative_logits = args.gamma - tf.reduce_sum(tf.abs(nh + nr - nt), axis=-1)
            positive_loss = tf.reduce_mean(tf.nn.softplus(-positive_logits))
            negative_losses = tf.nn.softplus(negative_logits)
            if args.adversarial:
                weights = tf.stop_gradient(tf.nn.softmax(
                    negative_logits * args.adversarial_temperature, axis=1))
                negative_loss = tf.reduce_mean(tf.reduce_sum(
                    weights * negative_losses, axis=1))
            else:
                negative_loss = tf.reduce_mean(negative_losses)
            loss = (positive_loss + negative_loss) * 0.5
        gradients = tape.gradient(loss, (entity_var, relation_var))
        optimizer.apply_gradients(zip(gradients, (entity_var, relation_var)))
        return loss

    for epoch in range(1, epochs + 1):
        ordering = rng.permutation(len(triples))
        losses = []
        batches = range(0, len(ordering), args.batch_size)
        with tqdm(total=len(batches), desc="TransE {}/{}".format(epoch, epochs),
                  unit="batch", dynamic_ncols=True) as progress:
            for offset in batches:
                positive = triples[ordering[offset:offset + args.batch_size]]
                negative = sample_negatives(positive, n_entities, n_relations,
                                            args.negatives, known_keys, rng)
                loss = float(step(tf.convert_to_tensor(positive, dtype=tf.int32),
                                  tf.convert_to_tensor(negative, dtype=tf.int32)).numpy())
                losses.append(loss)
                progress.update()
                progress.set_postfix(loss="{:.4f}".format(np.mean(losses)), refresh=False)
        # TransE's entity constraint is separate from the optimizer objective.
        entity_var.assign(tf.math.l2_normalize(entity_var, axis=1))
        metrics = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        should_evaluate = (valid_triples is not None and
                           (epoch % args.eval_every == 0 or epoch == epochs))
        if should_evaluate:
            valid = filtered_link_prediction(entity_var.numpy(), relation_var.numpy(),
                                             valid_triples, filter_triples,
                                             max_examples=args.eval_samples, seed=seed)
            metrics.update({"valid_" + key: value for key, value in valid.items()})
            if best is None or valid["mrr"] > best["valid"]["mrr"]:
                best = {"epoch": epoch, "valid": valid,
                        "entities": entity_var.numpy().copy(),
                        "relations": relation_var.numpy().copy()}
                patience_count = 0
            else:
                patience_count += 1
            print("TransE epoch {} loss {:.4f} val MRR {:.4f} Hits@1 {:.4f} Hits@10 {:.4f}".format(
                epoch, metrics["train_loss"], valid["mrr"], valid["hits_at_1"],
                valid["hits_at_10"]), flush=True)
        else:
            print("TransE epoch {} loss {:.4f}".format(
                epoch, metrics["train_loss"]), flush=True)
        logs.append(metrics)
        if valid_triples is not None and args.patience > 0 and patience_count >= args.patience:
            print("Stopping after {} validation checks without MRR improvement".format(
                args.patience), flush=True)
            break

    if best is None:
        best = {"epoch": epochs, "valid": None,
                "entities": entity_var.numpy().copy(),
                "relations": relation_var.numpy().copy()}
    return best, logs


def save_export(path, entity_names, relation_names, entity_vectors, relation_vectors):
    """Keep SRN's existing relation archive keys and add a separate entity archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, relations=np.asarray(relation_names, dtype=str),
                        vectors=np.asarray(relation_vectors, dtype=np.float32))
    entity_path = Path(str(path) + ".entities.npz")
    np.savez_compressed(entity_path, entities=np.asarray(entity_names, dtype=str),
                        vectors=np.asarray(entity_vectors, dtype=np.float32))
    return entity_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="Tab-separated head, relation, tail KG triples")
    parser.add_argument("--output", required=True, help="SRN-compatible relation NPZ")
    parser.add_argument("--dim", type=int, default=50,
                        help="Must remain 50 for the current SRN embedder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negatives", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=12.0)
    parser.add_argument("--adversarial", action="store_true",
                        help="Use self-adversarial negative weighting")
    parser.add_argument("--adversarial-temperature", type=float, default=1.0)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-samples", type=int, default=256,
                        help="Maximum held-out triples evaluated per split (0 evaluates all)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Validation checks without improvement (0 disables early stopping)")
    parser.add_argument("--refit-full", action="store_true",
                        help="Retrain on the entire KG for SRN export after held-out evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.dim != 50 or args.epochs < 1 or args.batch_size < 1 or
            args.negatives < 1 or args.eval_every < 1 or args.eval_samples < 0 or
            args.learning_rate <= 0 or args.gamma <= 0 or
            args.adversarial_temperature <= 0 or args.patience < 0):
        parser.error("Invalid hyperparameters (SRN requires --dim 50)")
    triples, entities, relations = read_triples(args.graph)
    train, valid, test = split_triples(triples, len(entities), len(relations),
                                       args.valid_ratio, args.test_ratio, args.seed)
    print("TransE: {} entities, {} relations, {} unique triples".format(
        len(entities), len(relations), len(triples)), flush=True)
    print("Link-prediction split: train={} valid={} test={}".format(
        len(train), len(valid), len(test)), flush=True)
    best, history = train_transe(train, len(entities), len(relations), args,
                                  epochs=args.epochs, seed=args.seed,
                                  valid_triples=valid, filter_triples=triples)
    test_metrics = filtered_link_prediction(
        best["entities"], best["relations"], test, triples,
        max_examples=args.eval_samples, seed=args.seed + 1)
    print("Best epoch {} | held-out test filtered MRR {:.4f} Hits@1 {:.4f} Hits@3 {:.4f} Hits@10 {:.4f}".format(
        best["epoch"], test_metrics["mrr"], test_metrics["hits_at_1"],
        test_metrics["hits_at_3"], test_metrics["hits_at_10"]), flush=True)

    export_model = best
    if args.refit_full:
        print("Refitting TransE on the full graph for {} epochs...".format(best["epoch"]), flush=True)
        export_model, _ = train_transe(triples, len(entities), len(relations), args,
                                       epochs=best["epoch"], seed=args.seed)
    output = Path(args.output)
    entity_output = save_export(output, list(entities), list(relations),
                                export_model["entities"], export_model["relations"])
    history_path = Path(str(output) + ".training.csv")
    columns = ("epoch", "train_loss", "valid_triples_evaluated", "valid_predictions_evaluated",
               "valid_mrr", "valid_mr", "valid_hits_at_1", "valid_hits_at_3", "valid_hits_at_10")
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    report_path = Path(str(output) + ".metrics.json")
    report = {
        "graph": str(args.graph), "seed": args.seed,
        "entities": len(entities), "relations": len(relations),
        "triples": len(triples), "train_triples": len(train),
        "valid_triples": len(valid), "test_triples": len(test),
        "hyperparameters": {key: value for key, value in vars(args).items()
                            if key not in ("graph", "output")},
        "selection_metric": "validation filtered MRR",
        "best_epoch": best["epoch"], "validation": best["valid"],
        "test": test_metrics, "refit_full": args.refit_full,
        "evaluation_model": "held-out training partition",
        "export_model": ("full-graph refit" if args.refit_full else "validation-selected"),
        "export": {"relations": str(output), "entities": str(entity_output)},
        "note": "Held-out metrics describe the selected model, not the optional full-graph refit.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved {} relation vectors to {}".format(len(relations), output), flush=True)
    print("Entity vectors: {} | history: {} | metrics: {}".format(
        entity_output, history_path, report_path), flush=True)


if __name__ == "__main__":
    main()
