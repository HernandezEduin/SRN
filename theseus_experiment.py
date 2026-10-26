"""Train or evaluate SRN on the released Kinship and MQuAKE-ST splits."""

import argparse
import csv
import json
import random
from pathlib import Path

from theseus_data import load_release


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kinship", "mquake-st"), required=True)
    parser.add_argument("--hops", choices=("2", "3", "4", "n"), default="2")
    parser.add_argument("--answer-mode", choices=("single", "multi"), default="single")
    parser.add_argument("--mode", choices=("download", "train", "evaluate"), default="train")
    parser.add_argument("--data-root", default="datasets/theseus")
    parser.add_argument("--glove", default="datasets/embeddings/glove.6B.50d.txt")
    parser.add_argument("--relation-embeddings", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None, help="Required for evaluation")
    parser.add_argument("--oracle-hops", action="store_true", help="Explicitly allow using annotated hop lengths for mixed n-hop data")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: limit each split to N rows")
    return parser


def save_predictions(path, predictions):
    with path.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = argument_parser().parse_args()
    if args.hops == "n" and args.mode != "download" and not args.oracle_hops:
        raise SystemExit("Mixed-hop SRN requires annotated horizons. Pass --oracle-hops to acknowledge this non-comparable setting, or use --hops 2/3/4")
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
    graph_relations = {data["relation"] for _, _, data in graph.edges(data=True)}
    missing = graph_relations - set(embedder.relation_embedding)
    if missing:
        raise ValueError("Missing {} graph relation embeddings, e.g. {}".format(len(missing), sorted(missing)[:5]))
    max_hops = max(record[3] for rows in splits.values() for record in rows)
    policy = TheseusSRN(graph, embedder, max_hops, learning_rate=args.learning_rate,
                        beam_size=args.beam_size, max_actions=args.max_actions)
    checkpoint = tf.train.Checkpoint(model=policy, optimizer=policy.optimizer)
    manager = tf.train.CheckpointManager(checkpoint, str(output / "checkpoints"), max_to_keep=3)
    if args.checkpoint:
        checkpoint.restore(args.checkpoint).expect_partial()
    elif args.mode == "evaluate":
        if not manager.latest_checkpoint:
            raise SystemExit("No checkpoint found. Use --checkpoint or train first")
        checkpoint.restore(manager.latest_checkpoint).expect_partial()

    if args.mode == "train":
        metrics_file = output / "metrics.csv"
        with metrics_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("epoch", "train_hits1", "train_loss", "dev_hits1"))
            writer.writeheader()
            for epoch in range(args.epochs):
                train_records = splits["train"].copy()
                random.shuffle(train_records)
                hits, losses = 0, []
                for number, record in enumerate(train_records, 1):
                    prediction, loss, _ = policy.train_query(record)
                    hits += int(prediction in record[2])
                    if loss is not None:
                        losses.append(loss)
                    if number % 1000 == 0:
                        print("Epoch {} trained {}/{}".format(epoch + 1, number, len(train_records)), flush=True)
                dev_hits1, dev_predictions = policy.evaluate(splits["dev"])
                metrics = {"epoch": epoch + 1,
                           "train_hits1": hits / len(train_records) if train_records else 0,
                           "train_loss": float(np.mean(losses)) if losses else 0,
                           "dev_hits1": dev_hits1}
                writer.writerow(metrics)
                handle.flush()
                manager.save(checkpoint_number=epoch + 1)
                save_predictions(output / "dev_predictions.jsonl", dev_predictions)
                print(metrics, flush=True)

    # Test is held out from epoch selection. Evaluation uses only model scores.
    for split in ("dev", "test"):
        hits1, predictions = policy.evaluate(splits[split])
        save_predictions(output / (split + "_predictions.jsonl"), predictions)
        print("{} Hits@1 {:.4f} on {} examples".format(split, hits1, len(predictions)))
    print("Missing GloVe tokens: {}".format(len(embedder.missing_words)))


if __name__ == "__main__":
    main()
