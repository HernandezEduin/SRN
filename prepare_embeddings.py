"""Optional TransE pretraining for SRN's frozen relation inputs.

This is independent of SRN's reinforcement-learning training. No QA labels,
questions or reference paths are read. External 50d TransE vectors also work.
"""

import argparse
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
            h, r, t = parts
            for entity in (h, t):
                if entity not in entities:
                    entities[entity] = len(entities)
            if r not in relations:
                relations[r] = len(relations)
            triples.append((entities[h], relations[r], entities[t]))
    if not triples:
        raise ValueError("Graph contains no triples")
    return np.asarray(triples, dtype=np.int32), entities, relations


def pretrain(triples, n_entities, n_relations, dim=50, epochs=100, batch_size=2048,
             learning_rate=0.01, margin=1.0, seed=42):
    """TransE with L1 distance, margin-ranking loss and head/tail corruption."""
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


def main():
    parser = argparse.ArgumentParser(description="Pretrain frozen SRN relation vectors")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    triples, entities, relations = read_triples(args.graph)
    vectors = pretrain(triples, len(entities), len(relations), epochs=args.epochs,
                       batch_size=args.batch_size, learning_rate=args.learning_rate,
                       margin=args.margin, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, relations=np.asarray(list(relations)), vectors=vectors)
    print("Saved {} relation embeddings to {}".format(len(relations), output))


if __name__ == "__main__":
    main()
