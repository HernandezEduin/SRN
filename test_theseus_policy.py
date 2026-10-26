"""Small policy smoke test requiring TensorFlow CPU."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import networkx as nx

from theseus_policy import TheseusSRN
from components.Embedder import Embedder
from theseus_data import load_graph, load_questions


class StubEmbeddings:
    def __init__(self):
        self.relations = {key: np.eye(50, dtype=np.float32)[i] for i, key in enumerate(("p", "q", "r"))}

    def embed_word(self, word):
        return np.ones(50, dtype=np.float32) / 50

    def embed_relation(self, relation):
        return self.relations[relation]


class PolicySmokeTest(unittest.TestCase):
    def test_official_kinship_example_if_cached(self):
        graph_path = Path("datasets/theseus/kinship/kg/orig/triplets.txt")
        qa_path = Path("datasets/theseus/kinship/qa/kinship_qa_2hop.csv")
        if not graph_path.is_file() or not qa_path.is_file():
            self.skipTest("Real Kinship files have not been downloaded")
        graph = load_graph(graph_path)
        questions = load_questions(qa_path, graph, "kinship")

        class EveryRelation:
            def embed_word(self, word):
                return np.full(50, 0.01, dtype=np.float32)

            def embed_relation(self, relation):
                vector = np.zeros(50, dtype=np.float32)
                vector[sum(relation.encode("utf-8")) % 50] = 1.0
                return vector

        model = TheseusSRN(graph, EveryRelation(), max_hops=2, max_actions=200)
        example = questions["train"][0]
        prediction, loss, _ = model.train_query(example)
        self.assertTrue(graph.has_node(prediction))
        self.assertTrue(np.isfinite(loss))
        endpoint, path = model.predict_query(example)
        self.assertEqual(endpoint, path[-1][2])

    def test_embedding_file_and_checkpoint(self):
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as temp:
            glove = Path(temp) / "tiny_glove.txt"
            glove.write_text("where " + " ".join(["0.1"] * 50) + "\n", encoding="utf-8")
            relation_path = Path(temp) / "relations.npz"
            np.savez_compressed(relation_path, relations=np.asarray(["p"]),
                                vectors=np.ones((1, 50), dtype=np.float32))
            embedding = Embedder(glove, relation_path, vocabulary={"where", "missing"})
            self.assertEqual(embedding.embed_word("missing").shape, (50,))
            graph = nx.MultiDiGraph()
            graph.add_edge("A", "B", relation="p")
            model = TheseusSRN(graph, embedding, max_hops=1)
            model.train_query((["where"], "A", frozenset(("B",)), 1, "q"))
            checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
            saved = checkpoint.save(str(Path(temp) / "checkpoint"))
            self.assertTrue(Path(saved + ".index").is_file())

    def test_sampled_reinforce_and_answer_blind_beam(self):
        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", relation="p")
        graph.add_edge("B", "C", relation="q")
        graph.add_edge("B", "D", relation="r")
        model = TheseusSRN(graph, StubEmbeddings(), max_hops=2, beam_size=2, max_actions=10)
        record = (["where", "<e>"], "A", frozenset(("C",)), 2, "toy")
        destination, loss, _ = model.train_query(record)
        self.assertIn(destination, ("C", "D"))
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(len(model.trainable_variables), 0)
        actual, path = model.predict_query(record)
        other, _ = model.predict_query((record[0], "A", frozenset(("D",)), 2, "toy"))
        self.assertEqual(actual, other)
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], ("A", "p", "B"))


if __name__ == "__main__":
    unittest.main()
