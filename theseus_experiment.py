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


def main():
    args = argument_parser().parse_args()
    if args.progress_seconds < 0:
        raise SystemExit("--progress-seconds must be non-negative")
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

                dev_started = time.monotonic()
                with ProgressReporter(
                    "Epoch {}/{} dev".format(epoch + 1, args.epochs),
                    len(splits["dev"]), args.progress_seconds, args.progress_style
                ) as progress:
                    dev_hits1, dev_predictions = policy.evaluate(
                        ProgressRecords(splits["dev"], progress))
                dev_seconds = time.monotonic() - dev_started

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
    print("GloVe OOV tokens encountered during rollout (excluding <e>): {}".format(
        len(embedder.missing_words - {"<e>"})))


if __name__ == "__main__":
    main()
