"""Train or evaluate SRN on the released Kinship and MQuAKE-ST splits."""

import argparse
import csv
import json
import random
import sys
import threading
import time
from pathlib import Path

from tqdm.auto import tqdm

from glove_coverage import compute_glove_coverage, format_glove_coverage
from theseus_data import load_release


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kinship", "mquake-st"), required=True)
    parser.add_argument("--hops", choices=("2", "3", "4", "n"), default="n")
    parser.add_argument("--answer-mode", choices=("single", "multi"), default="single")
    parser.add_argument("--mode", choices=("download", "train", "evaluate", "diagnose"), default="train")
    parser.add_argument("--data-root", default="datasets/theseus")
    parser.add_argument("--glove", default="datasets/embeddings/glove.6B.50d.txt")
    parser.add_argument("--relation-embeddings", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--dev-every", type=int, default=1,
                        help="Evaluate on dev every N epochs and on the final epoch (default: 1)")
    parser.add_argument("--sensitivity-pairs", type=int, default=0,
                        help="Compare question representations and per-hop action distributions for N pairs sharing a source")
    parser.add_argument("--gradient-samples", type=int, default=0,
                        help="During diagnose, inspect REINFORCE gradient norms for N sampled training questions without updating weights")
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--max-hops", type=int, default=None,
                        help="Global rollout horizon T (default: maximum hop length in selected split). "
                             "Per-question annotated hops are never used to control rollout.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Required for evaluation")
    parser.add_argument("--oracle-hops", action="store_true",
                        help="Deprecated and ignored. SRN now learns STOP without annotated hop horizons.")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: limit each split to N rows")
    parser.add_argument("--progress-seconds", type=float, default=15.0,
                        help="Progress heartbeat frequency in seconds (0 disables periodic heartbeat)")
    parser.add_argument("--progress-style", choices=("auto", "tqdm", "plain"), default="auto",
                        help="Display tqdm in a terminal, or periodic plain-text logs otherwise")
    return parser


def format_duration(seconds):
    """Format a positive duration as hours:minutes:seconds."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


class ProgressReporter:
    """Display a tqdm bar in terminals and a heartbeat in non-interactive logs."""

    def __init__(self, label, total, interval_seconds, style="auto"):
        self.label = label
        self.total = total
        self.interval_seconds = interval_seconds
        self.completed = 0
        self.started = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._bar = None
        if style not in ("auto", "tqdm", "plain"):
            raise ValueError("Unknown progress style: {}".format(style))
        self._use_tqdm = sys.stderr.isatty() if style == "auto" else style == "tqdm"

    def __enter__(self):
        self.started = time.monotonic()
        if self._use_tqdm:
            self._bar = tqdm(total=self.total, desc=self.label, unit="query",
                             dynamic_ncols=True, mininterval=0.2, file=sys.stderr)
        else:
            print("{}: 0/{} (0.0%) | elapsed 00:00:00 | ETA estimating".format(
                self.label, self.total), flush=True)
        if self.interval_seconds > 0:
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(self.interval_seconds):
            self.report()

    def report(self):
        with self._lock:
            if self._bar is not None:
                # Refresh independently from the worker. This updates elapsed
                # time even when one query is blocked by GPU JIT compilation.
                self._bar.refresh()
                return
            elapsed = time.monotonic() - self.started
            percent = 100.0 * self.completed / self.total if self.total else 100.0
            eta = (format_duration(elapsed / self.completed * (self.total - self.completed))
                   if self.completed else "estimating")
            print("{}: {}/{} ({:.1f}%) | elapsed {} | ETA {}".format(
                self.label, self.completed, self.total, percent,
                format_duration(elapsed), eta), flush=True)

    def update(self, count, **metrics):
        with self._lock:
            self.completed = count
            if self._bar is not None:
                if metrics:
                    self._bar.set_postfix(metrics, refresh=False)
                self._bar.update(count - self._bar.n)
                if count == self.total:
                    self._bar.refresh()
                return
        if count == 1 or count == self.total:
            self.report()

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._bar is not None:
            self._bar.close()
        if exc_type is not None:
            print("{} interrupted at {}/{}".format(
                self.label, self.completed, self.total), flush=True)


class ProgressRecords:
    """Wrap evaluation records without changing the policy's evaluation path."""

    def __init__(self, records, reporter):
        self.records = records
        self.reporter = reporter

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        for count, record in enumerate(self.records, 1):
            yield record
            self.reporter.update(count)


def save_predictions(path, predictions):
    with path.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def question_sensitivity(policy, records, count):
    """Compare questions on a shared, gold-blind greedy navigation prefix.

    At each hop both questions use the SAME graph node and GRU history, so
    policy differences can only arise from the question representation.
    The prefix is chosen by the first question and never from gold paths.
    """
    import numpy as np
    import tensorflow as tf
    from Environment import STOP_RELATION

    first_by_source, used_sources, pairs = {}, set(), []
    for record in records:
        source = record[1]
        if source in used_sources:
            continue
        prior = first_by_source.get(source)
        if prior is None:
            first_by_source[source] = record
        elif prior[0] != record[0]:
            pairs.append((prior, record))
            used_sources.add(source)
            if len(pairs) >= count:
                break

    def representation_comparison(first_vectors, second_vectors):
        first_vector = np.mean(np.asarray(first_vectors), axis=0)
        second_vector = np.mean(np.asarray(second_vectors), axis=0)
        difference = first_vector - second_vector
        first_norm = np.linalg.norm(first_vector)
        second_norm = np.linalg.norm(second_vector)
        cosine = (float(np.dot(first_vector, second_vector) /
                        (first_norm * second_norm))
                  if first_norm > 1e-12 and second_norm > 1e-12 else None)
        return {"cosine_similarity": cosine,
                "pooled_l2_distance": float(np.linalg.norm(difference))}

    results = []
    for first, second in pairs:
        source = first[1]
        first_words = np.stack([policy.Embedder.embed_word(w) for w in first[0]])
        second_words = np.stack([policy.Embedder.embed_word(w) for w in second[0]])
        encoded = [
            policy.encode(first[0], training=False),
            policy.encode(second[0], training=False),
        ]
        encodings = {
            "input_embeddings": representation_comparison(first_words, second_words),
            "BiGRU": representation_comparison(encoded[0].numpy(), encoded[1].numpy()),
        }

        node, history, states = source, tf.zeros(policy.Embedder.dim, tf.float32), None
        per_hop, greedy_prefix = [], []
        for hop in range(1, policy.max_hops + 1):
            actions = [(data["relation"], target)
                       for _, target, data in policy.graph.out_edges(node, data=True)]
            actions.append((STOP_RELATION, node))
            # Both questions must use the same outgoing actions at each hop.
            hop_questions = [policy.SLP(value, t=hop) for value in encoded]
            distributions = [
                tf.nn.softmax(policy.action_scores(question, history, actions)).numpy()
                for question in hop_questions
            ]
            probability_difference = np.abs(distributions[0] - distributions[1])
            best = [int(np.argmax(distribution)) for distribution in distributions]
            entropy = [
                float(-np.sum(distribution * np.log(
                    np.maximum(distribution, 1e-30))))
                for distribution in distributions
            ]
            per_hop.append({
                "hop": hop,
                "node": node,
                "candidate_edges": len(actions),
                "total_variation": float(0.5 * probability_difference.sum()),
                "max_probability_difference": float(probability_difference.max()),
                "same_top_action": best[0] == best[1],
                "top_actions": [list(actions[index]) for index in best],
                "top_action_probabilities": [
                    float(distributions[index][best[index]]) for index in (0, 1)],
                "action_entropy": entropy,
                "stop_probabilities": [
                    float(distribution[-1]) for distribution in distributions],
                "SLP": representation_comparison(
                    hop_questions[0].numpy(), hop_questions[1].numpy()),
            })
            relation, target = actions[best[0]]
            if relation == STOP_RELATION:
                break
            greedy_prefix.append([node, relation, target])
            if hop < policy.max_hops:
                history, states = policy.GRU(
                    policy.relation_vector(relation), states=states, training=False)
            node = target

        first_hop = per_hop[0]
        results.append({
            "source": source,
            "question_ids": [first[4], second[4]],
            "question_tokens": [first[0], second[0]],
            "candidate_edges": first_hop["candidate_edges"],
            "total_variation": first_hop["total_variation"],
            "max_probability_difference": first_hop["max_probability_difference"],
            "same_top_action": first_hop["same_top_action"],
            "top_actions": first_hop["top_actions"],
            "stop_probabilities": first_hop["stop_probabilities"],
            "representations": encodings,
            "greedy_prefix": greedy_prefix,
            "per_hop": per_hop,
        })

    by_hop = []
    for hop in range(1, policy.max_hops + 1):
        rows = [row["per_hop"][hop - 1] for row in results
                if len(row["per_hop"]) >= hop]
        if rows:
            by_hop.append({
                "hop": hop,
                "pairs": len(rows),
                "mean_total_variation": float(np.mean(
                    [row["total_variation"] for row in rows])),
                "same_top_action_rate": float(np.mean(
                    [row["same_top_action"] for row in rows])),
                "mean_stop_probability": [
                    float(np.mean([row["stop_probabilities"][i] for row in rows]))
                    for i in (0, 1)],
            })
    return {
        "requested_pairs": count,
        "pairs_evaluated": len(results),
        "mean_total_variation": float(np.mean(
            [row["total_variation"] for row in results]
        )) if results else None,
        "same_top_action_rate": float(np.mean(
            [row["same_top_action"] for row in results]
        )) if results else None,
        "by_hop": by_hop,
        "warning": "Gold-blind question comparison. All paired distributions "
                   "share the same source, candidate edges, and greedy prefix. "
                   "The prefix follows the first question, not reference paths.",
        "pairs": results,
    }


def write_question_sensitivity(policy, records, count, output):
    report = question_sensitivity(policy, records, count)
    path = output / "question_sensitivity.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print("Question sensitivity: {} pairs, mean first-hop total variation {}, "
          "same top action rate {}. Per-hop details: {}".format(
              report["pairs_evaluated"], report["mean_total_variation"],
              report["same_top_action_rate"], path), flush=True)


def write_gradient_diagnostics(policy, records, count, output, seed,
                               progress_seconds=15.0, progress_style="auto"):
    """Measure actual sampled REINFORCE gradients without optimizer updates."""
    import numpy as np

    selected = random.Random(seed).sample(records, min(count, len(records)))
    samples = []
    with ProgressReporter("REINFORCE gradient diagnostic", len(selected),
                          progress_seconds, progress_style) as progress:
        for number, record in enumerate(selected, 1):
            prediction, loss, path, statistics = policy.train_query(
                record, update=False, report_gradients=True)
            samples.append({
                "id": record[4],
                "correct": bool(prediction in record[2]),
                "loss": loss,
                "path_edge_count": len(path),
                **statistics,
            })
            progress.update(number)

    names = ("BiGRU", "SLP", "Attention", "Perceptron", "GRU", "STOP")
    summary = {}
    for name in names:
        values = [sample["groups"][name]["l2"] for sample in samples
                  if name in sample["groups"]]
        summary[name] = {
            "mean_gradient_l2": float(np.mean(values)) if values else None,
            "nonzero_fraction": float(np.mean([value > 1e-12 for value in values]))
            if values else None,
        }
    report = {
        "scope": "Sampled on-policy SRN REINFORCE loss on training QA, without optimizer updates",
        "requested_samples": count,
        "samples_evaluated": len(samples),
        "mean_terminal_success": float(np.mean([sample["correct"] for sample in samples]))
        if samples else None,
        "groups": summary,
        "samples": samples,
    }
    path = output / "gradient_diagnostics.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print("Gradient diagnostics: {} training queries. Details: {}".format(
        len(samples), path), flush=True)


def main():
    args = argument_parser().parse_args()
    if args.progress_seconds < 0:
        raise SystemExit("--progress-seconds must be non-negative")
    if args.dev_every < 1 or args.sensitivity_pairs < 0 or args.gradient_samples < 0 or args.beam_size < 1:
        raise SystemExit("--dev-every and --beam-size must be positive; diagnostic counts must be non-negative")
    if args.max_actions < 2:
        raise SystemExit("--max-actions must be at least 2 to retain STOP alongside KG edges")
    if args.max_hops is not None and args.max_hops < 1:
        raise SystemExit("--max-hops must be positive")
    if args.mode == "diagnose" and args.sensitivity_pairs == 0 and args.gradient_samples == 0:
        raise SystemExit("--mode diagnose requires --sensitivity-pairs N or --gradient-samples N")
    if args.gradient_samples and args.mode != "diagnose":
        raise SystemExit("--gradient-samples is only supported with --mode diagnose")
    if args.oracle_hops:
        print("Warning: --oracle-hops is deprecated and ignored. "
              "Rollouts use fixed T with learned STOP.", flush=True)
    graph, splits, graph_path = load_release(args.dataset, args.hops, args.answer_mode, args.data_root)
    print("Graph: {} nodes, {} directed triples. Splits: {}".format(
        graph.number_of_nodes(), graph.number_of_edges(),
        {split: len(rows) for split, rows in splits.items()}), flush=True)
    if args.mode == "download":
        print("Graph file: {}".format(graph_path))
        return

    # Import tensorflow only for training/evaluation, not for dataset preparation.
    import numpy as np
    import tensorflow as tf
    from components.Embedder import Embedder
    from theseus_policy import TheseusSRN

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        splits = {split: rows[:args.limit] for split, rows in splits.items()}
    output = Path(args.output or "saved_models/theseus_{}_{}_{}".format(args.dataset, args.answer_mode, args.hops))
    output.mkdir(parents=True, exist_ok=True)
    relation_path = args.relation_embeddings or str(output / "transe_relations.npz")
    vocabulary = {word for rows in splits.values() for record in rows for word in record[0]}
    embedder = Embedder(args.glove, relation_path, vocabulary)
    glove_report = compute_glove_coverage(splits, embedder.word_embeddings)
    print(format_glove_coverage(glove_report, embedder.dim), flush=True)
    coverage_file = output / "glove_coverage.json"
    coverage_file.write_text(json.dumps(glove_report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print("GloVe coverage saved to {}".format(coverage_file), flush=True)
    graph_relations = {data["relation"] for _, _, data in graph.edges(data=True)}
    missing = graph_relations - set(embedder.relation_embedding)
    if missing:
        raise ValueError("Missing {} graph relation embeddings, e.g. {}".format(len(missing), sorted(missing)[:5]))
    # The dataset hop column is used only to choose a dataset-level default T.
    # It must never control an individual train/eval rollout.
    max_hops = args.max_hops if args.max_hops is not None else max(
        record[3] for rows in splits.values() for record in rows)
    print("SRN rollout horizon T={} with learned STOP (no gold-hop oracle)".format(
        max_hops), flush=True)
    policy = TheseusSRN(graph, embedder, max_hops, learning_rate=args.learning_rate,
                        beam_size=args.beam_size, max_actions=args.max_actions)
    # Keras 3 may mark an RNN as built while restoring a deferred checkpoint,
    # before its GRUCell variables exist. Materialize every layer and optimizer
    # slot first, for train, evaluate, and diagnose alike.
    policy._initialize_optimizer()
    checkpoint = tf.train.Checkpoint(model=policy, optimizer=policy.optimizer)
    manager = tf.train.CheckpointManager(checkpoint, str(output / "checkpoints"), max_to_keep=3)
    if args.checkpoint:
        checkpoint.restore(args.checkpoint).expect_partial()
    elif args.mode in ("evaluate", "diagnose"):
        if not manager.latest_checkpoint:
            raise SystemExit("No checkpoint found. Use --checkpoint or train first")
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

    if args.mode == "diagnose":
        if args.sensitivity_pairs:
            write_question_sensitivity(policy, splits["dev"], args.sensitivity_pairs, output)
        if args.gradient_samples:
            write_gradient_diagnostics(
                policy, splits["train"], args.gradient_samples, output, args.seed,
                args.progress_seconds, args.progress_style)
        return

    if args.mode == "train":
        metrics_file = output / "metrics.csv"
        fields = ("epoch", "train_hits1", "train_loss", "dev_hits1",
                  "train_seconds", "dev_seconds", "epoch_seconds")
        with metrics_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(args.epochs):
                epoch_started = time.monotonic()
                train_records = splits["train"].copy()
                random.shuffle(train_records)
                hits, losses, running_loss = 0, [], 0.0
                with ProgressReporter(
                    "Epoch {}/{} training".format(epoch + 1, args.epochs),
                    len(train_records), args.progress_seconds, args.progress_style
                ) as progress:
                    for number, record in enumerate(train_records, 1):
                        prediction, loss, _ = policy.train_query(record)
                        hits += int(prediction in record[2])
                        if loss is not None:
                            losses.append(loss)
                            running_loss += loss
                        progress.update(
                            number, loss="{:.4f}".format(running_loss / len(losses)) if losses else "n/a",
                            hits1="{:.4f}".format(hits / number))
                train_seconds = time.monotonic() - epoch_started

                run_dev = (epoch + 1) % args.dev_every == 0 or epoch + 1 == args.epochs
                dev_hits1, dev_predictions, dev_seconds = "", None, 0.0
                if run_dev:
                    dev_started = time.monotonic()
                    with ProgressReporter(
                        "Epoch {}/{} dev".format(epoch + 1, args.epochs),
                        len(splits["dev"]), args.progress_seconds, args.progress_style
                    ) as progress:
                        dev_hits1, dev_predictions = policy.evaluate(
                            ProgressRecords(splits["dev"], progress))
                    dev_seconds = time.monotonic() - dev_started
                else:
                    print("Epoch {}/{}: skipping dev (--dev-every {})".format(
                        epoch + 1, args.epochs, args.dev_every), flush=True)

                metrics = {"epoch": epoch + 1,
                           "train_hits1": hits / len(train_records) if train_records else 0,
                           "train_loss": float(np.mean(losses)) if losses else 0,
                           "dev_hits1": dev_hits1,
                           "train_seconds": round(train_seconds, 3),
                           "dev_seconds": round(dev_seconds, 3),
                           "epoch_seconds": round(time.monotonic() - epoch_started, 3)}
                writer.writerow(metrics)
                handle.flush()
                print("Epoch {}/{} saving checkpoint...".format(
                    epoch + 1, args.epochs), flush=True)
                manager.save(checkpoint_number=epoch + 1)
                if dev_predictions is not None:
                    save_predictions(output / "dev_predictions.jsonl", dev_predictions)
                print("Epoch {}/{} completed: {}".format(
                    epoch + 1, args.epochs, metrics), flush=True)

    # Test is held out from epoch selection. Evaluation uses only model scores.
    for split in ("dev", "test"):
        with ProgressReporter(
            "Final {} evaluation".format(split), len(splits[split]),
            args.progress_seconds, args.progress_style
        ) as progress:
            hits1, predictions = policy.evaluate(
                ProgressRecords(splits[split], progress))
        save_predictions(output / (split + "_predictions.jsonl"), predictions)
        print("{} Hits@1 {:.4f} on {} examples".format(
            split, hits1, len(predictions)), flush=True)
    if args.sensitivity_pairs:
        write_question_sensitivity(policy, splits["dev"], args.sensitivity_pairs, output)
    print("GloVe OOV tokens encountered during rollout (excluding <e>): {}".format(
        len(embedder.missing_words - {"<e>"})))


if __name__ == "__main__":
    main()
