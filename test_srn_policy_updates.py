"""Regression checks for paper-aligned SRN rewards and batched scoring."""

import unittest

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is required for SRN policy tests")
class SRNPolicyUpdatesTests(unittest.TestCase):
    def test_base_reward_depends_on_final_answer_only(self):
        from Environment import Rewards, State

        rewarder = Rewards(gamma=0.95)
        self.assertEqual(rewarder.terminal_base_reward("B", {"B"}), 1.0)
        self.assertEqual(rewarder.terminal_base_reward("C", {"B"}), 0.0)

        # Visiting the gold entity at an intermediate step must not produce a
        # base reward. This test isolates the shaping contribution at phi=0.
        first = State(None, "A", "A")
        intermediate = State(None, "A", "B")
        shaped = rewarder.get_transition_reward(
            first, ("related", "B"), intermediate, {"B"}
        )
        self.assertAlmostEqual(float(shaped.numpy()), 0.0)

    def test_batched_attention_and_scores_match_scalar_equations(self):
        from components.Attention import Attention
        from components.Perceptron import Perceptron

        tf.random.set_seed(42)
        relation_vectors = tf.random.normal((7, 50))
        question_vectors = tf.random.normal((8, 50))
        history = tf.random.normal((50,))
        attention = Attention()
        perceptron = Perceptron()
        expected = tf.stack([
            perceptron(relation, history, attention(relation, question_vectors))
            for relation in tf.unstack(relation_vectors)
        ])
        actual = perceptron.call_many(
            relation_vectors, history,
            attention.call_many(relation_vectors, question_vectors),
        )
        np.testing.assert_allclose(actual.numpy(), expected.numpy(),
                                   rtol=1e-5, atol=1e-5)

        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(perceptron.call_many(
                relation_vectors, history,
                attention.call_many(relation_vectors, question_vectors),
            ))
        weights = attention.trainable_variables + perceptron.trainable_variables
        grads = tape.gradient(loss, weights)
        self.assertTrue(weights)
        self.assertTrue(all(gradient is not None for gradient in grads))

    def test_beam_search_preserves_scalar_reference_paths(self):
        import networkx as nx

        from Environment import d
        from theseus_policy import TheseusSRN

        class FakeEmbedder:
            dim = d

            def embed_word(self, word):
                return np.full(d, (len(word) + 1) / 20.0, dtype=np.float32)

            def embed_relation(self, relation):
                return np.full(d, (len(relation) + 1) / 20.0, dtype=np.float32)

        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", relation="mother")
        graph.add_edge("A", "C", relation="father")
        graph.add_edge("B", "A", relation="daughter")
        graph.add_edge("B", "D", relation="sister")
        graph.add_edge("C", "D", relation="brother")
        graph.add_edge("C", "A", relation="son")
        policy = TheseusSRN(graph, FakeEmbedder(), max_hops=2, beam_size=3)
        record = (["who", "is", "the", "mother"], "A", {"D"}, 2, "q")

        # Reference implementation updates GRU states before beam pruning.
        def reference_prediction():
            from Environment import State
            encoded = policy.encode(record[0], training=False)
            questions = {t: policy.SLP(encoded, t=t) for t in (1, 2)}
            beams = [("A", 0.0, tf.zeros(d, tf.float32), None, [])]
            for step in (1, 2):
                candidates = []
                for node, path_score, history, states, path in beams:
                    policy.env.start_new_query(
                        State(encoded, "A", node), frozenset()
                    )
                    actions = policy.env.get_possible_actions()
                    if not actions:
                        candidates.append((node, path_score, history, states, path))
                        continue
                    logits = policy.action_scores(questions[step], history, actions)
                    actions, logits = policy.prune(actions, logits, policy.max_actions)
                    log_prob = tf.nn.log_softmax(logits).numpy()
                    for index in np.argsort(-log_prob)[:policy.beam_size]:
                        relation, target = actions[int(index)]
                        vector = tf.convert_to_tensor(
                            policy.Embedder.embed_relation(relation), tf.float32
                        )
                        next_history, next_states = policy.GRU(
                            vector, states=states, training=False
                        )
                        candidates.append((
                            target, path_score + float(log_prob[index]),
                            next_history, next_states,
                            path + [(node, relation, target)]
                        ))
                beams = sorted(candidates, key=lambda row: row[1],
                               reverse=True)[:policy.beam_size]
            return beams[0][0], beams[0][4]

        self.assertEqual(policy.predict_query(record), reference_prediction())


if __name__ == "__main__":
    unittest.main()
