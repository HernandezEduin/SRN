"""Run with python -m unittest test_theseus_data.py."""

import csv
import tempfile
import unittest
from pathlib import Path

from theseus_data import load_graph, load_questions, parse_answers, question_tokens, release_paths
from prepare_embeddings import pretrain, read_triples


class TheseusDataTests(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(release_paths("kinship", "n"),
                         ("kg/orig/triplets.txt", "qa/kinship_qa_nhop.csv"))
        self.assertEqual(release_paths("mquake-st", "4", "multi"),
                         ("kg/triplets.txt", "qa/multi_answers/qa_4hop.csv"))

    def test_directed_parallel_relations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "triplets.txt"
            path.write_text("Q1\tP1\tQ2\nQ1\tP2\tQ2\nQ2\tP3\tQ3\n", encoding="utf-8")
            graph = load_graph(path)
            self.assertEqual(graph.number_of_edges(), 3)
            self.assertFalse(graph.has_edge("Q2", "Q1"))
            self.assertEqual({edge["relation"] for edge in graph["Q1"]["Q2"].values()}, {"P1", "P2"})

    def test_gold_paths_not_required_and_splits_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            graph_path = Path(temp) / "triplets.txt"
            graph_path.write_text("Q1\tP1\tQ2\nQ1\tP2\tQ3\n", encoding="utf-8")
            graph = load_graph(graph_path)
            qa_path = Path(temp) / "qa.csv"
            with qa_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Question-Number", "Question", "Source",
                                                            "Source-Entity", "Answer-Entity", "Hops", "SplitLabel"])
                writer.writeheader()
                writer.writerow({"Question-Number": "1", "Question": "Who knows Alice?", "Source": "Alice",
                                 "Source-Entity": "Q1", "Answer-Entity": "['Q2', 'Q3']", "Hops": "2", "SplitLabel": "test"})
            splits = load_questions(qa_path, graph, "mquake-st")
            self.assertEqual(len(splits["train"]), 0)
            self.assertEqual(splits["test"][0][2], frozenset(("Q2", "Q3")))
            self.assertEqual(splits["test"][0][0], ["who", "knows", "<e>"])

    def test_answers_and_tokenization(self):
        self.assertEqual(parse_answers("Q17"), frozenset(("Q17",)))
        self.assertEqual(parse_answers("['Q17', 'Q18']"), frozenset(("Q17", "Q18")))
        self.assertEqual(question_tokens("Where is Hunter × Hunter from?", "Hunter × Hunter"),
                         ["where", "is", "<e>", "from"])

    def test_transe_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "kg.txt"
            path.write_text("A\tparent\tB\nB\tspouse\tC\n", encoding="utf-8")
            triples, entities, relations = read_triples(path)
            vectors = pretrain(triples, len(entities), len(relations), epochs=1, batch_size=2, seed=3)
            self.assertEqual(vectors.shape, (2, 50))
            self.assertTrue((abs(vectors) < 100).all())


if __name__ == "__main__":
    unittest.main()
