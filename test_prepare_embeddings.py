"""Fast, TensorFlow-free checks for TransE link-prediction preparation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_embeddings import (
    filtered_link_prediction,
    read_triples,
    sample_negatives,
    save_export,
    split_triples,
    triple_keys,
)


class PrepareEmbeddingsTests(unittest.TestCase):
    def test_split_is_disjoint_and_preserves_vocabulary(self):
        triples = np.asarray(
            [(head, relation, (head + delta) % 16)
             for head in range(16) for relation in range(2) for delta in (1, 2)],
            dtype=np.int64,
        )
        train, valid, test = split_triples(triples, 16, 2, 0.1, 0.1, 42)
        self.assertEqual(len(train) + len(valid) + len(test), len(triples))
        self.assertFalse({tuple(x) for x in train} & {tuple(x) for x in valid})
        self.assertFalse({tuple(x) for x in train} & {tuple(x) for x in test})
        self.assertFalse({tuple(x) for x in valid} & {tuple(x) for x in test})
        self.assertEqual(len(np.unique(np.r_[train[:, 0], train[:, 2]])), 16)
        self.assertEqual(len(np.unique(train[:, 1])), 2)

    def test_negatives_are_not_known_positives(self):
        positives = np.asarray([[0, 0, 1], [1, 0, 2], [2, 1, 3]], dtype=np.int64)
        keys = np.unique(triple_keys(positives, 5, 2))
        negatives = sample_negatives(
            positives, 5, 2, 24, keys, np.random.default_rng(42)
        )
        self.assertEqual(negatives.shape, (3, 24, 3))
        self.assertTrue(np.all(~np.isin(triple_keys(negatives, 5, 2), keys)))
        self.assertTrue(np.all(negatives[:, :, 1] == positives[:, None, 1]))

    def test_perfect_head_and_tail_ranking(self):
        entities = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
        relations = np.asarray([[1.0]], dtype=np.float32)
        triples = np.asarray([[0, 0, 1]], dtype=np.int64)
        metrics = filtered_link_prediction(entities, relations, triples, triples)
        self.assertEqual(metrics["predictions_evaluated"], 2)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["hits_at_1"], 1.0)

    def test_names_remain_aligned_with_vector_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "triplets.txt"
            graph.write_text("Alice\tfather\tBob\nAlice\tfather\tBob\n"
                             "Bob\tmother\tCarol\n", encoding="utf-8")
            triples, entities, relations = read_triples(graph)
            self.assertEqual(len(triples), 2)
            output = Path(directory) / "transe_relations.npz"
            entity_archive = save_export(
                output, list(entities), list(relations),
                np.arange(3 * 50, dtype=np.float32).reshape(3, 50),
                np.arange(2 * 50, dtype=np.float32).reshape(2, 50),
            )
            with np.load(output, allow_pickle=False) as loaded:
                mapping = dict(zip(loaded["relations"].astype(str),
                                   loaded["vectors"]))
                self.assertEqual(loaded["vectors"].shape, (2, 50))
                np.testing.assert_array_equal(
                    mapping["mother"], np.arange(50, 100, dtype=np.float32)
                )
            with np.load(entity_archive, allow_pickle=False) as loaded:
                self.assertEqual(set(loaded.files), {"entities", "vectors"})
                self.assertEqual(loaded["vectors"].shape, (3, 50))


if __name__ == "__main__":
    unittest.main()
