"""Preflight GloVe coverage tests, independent of TensorFlow."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from components.Embedder import Embedder
from glove_coverage import compute_glove_coverage, format_glove_coverage


class GloveCoverageTests(unittest.TestCase):
    def test_split_token_and_occurrence_coverage(self):
        splits = {
            "train": [(["Who", "is", "<e>", "oov"], "A", {"B"}, 2, "1"),
                      (["who", "oov"], "A", {"B"}, 2, "2")],
            "dev": [(["king", "oov"], "A", {"B"}, 2, "3")],
            "test": [],
        }
        vectors = {"who": [1.0], "is": [1.0], "king": [1.0]}
        report = compute_glove_coverage(splits, vectors)
        overall = report["all"]
        self.assertEqual(overall["unique_tokens"], 4)
        self.assertEqual(overall["matched_unique"], 3)
        self.assertEqual(overall["token_occurrences"], 7)
        self.assertEqual(overall["matched_occurrences"], 4)
        self.assertEqual(overall["excluded_special_tokens"], {"<e>": 1})
        self.assertEqual(overall["missing_tokens"],
                         [{"token": "oov", "occurrences": 3}])
        self.assertEqual(report["by_split"]["train"]["matched_occurrences"], 3)
        self.assertEqual(report["by_split"]["dev"]["matched_occurrences"], 1)
        self.assertEqual(report["by_split"]["test"]["unique_tokens"], 0)
        self.assertIn("unique 3/4", format_glove_coverage(report, 50))

    def test_real_embedder_uses_only_loaded_words_and_skips_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glove = root / "glove.txt"
            vector = " ".join(["0.25"] * 50)
            glove.write_text(
                "2 50\nwho {}\nknows {}\n".format(vector, vector),
                encoding="utf-8",
            )
            relations = root / "relations.npz"
            np.savez_compressed(
                relations,
                relations=np.asarray(["parent"]),
                vectors=np.ones((1, 50), dtype=np.float32),
            )
            embedder = Embedder(
                glove, relations, {"who", "knows", "<e>", "absent"}
            )
            report = compute_glove_coverage(
                {"train": [(["who", "knows", "<e>", "absent"], "", set(), 2, "q")]},
                embedder.word_embeddings,
            )
            self.assertEqual(report["all"]["matched_unique"], 2)
            self.assertEqual(report["all"]["unique_tokens"], 3)
            np.testing.assert_allclose(embedder.embed_word("WHO"),
                                       np.full(50, 0.25, dtype=np.float32))
            self.assertTrue(np.all(embedder.embed_word("absent") == 0))
            self.assertEqual(embedder.word_embeddings.keys(), {"who", "knows"})


if __name__ == "__main__":
    unittest.main()
