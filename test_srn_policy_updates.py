"""Regression checks for SRN's paper-aligned rewards, STOP and beam search."""

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
        intermediate = State(None, "A", "B")
        shaped = rewarder.get_transition_reward(
            State(None, "A", "A"), ("related", "B"), intermediate, {"B"}
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

    def test_stop_is_available_and_absorbing_even_at_sinks(self):
        import networkx as nx
        from Environment import Environment, Rewards, State, STOP_RELATION

        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", relation="family")
        environment = Environment(graph, allow_stop=True)
        environment.start_new_query(State(None, "A", "A"), {"A"})
        self.assertIn((STOP_RELATION, "A"), environment.get_possible_actions())
        state, _ = environment.transit(
            (STOP_RELATION, "A"), 1, {}, {1: tf.zeros(50)})
        self.assertTrue(state.terminal)
        self.assertEqual(state.e_t, "A")
        self.assertEqual(state.h_t, set())
        self.assertEqual(environment.get_possible_actions(), [])
        self.assertEqual(float(Rewards().phi(state).numpy()), 0.0)

        environment.start_new_query(State(None, "B", "B"), {"B"})
        self.assertEqual(environment.get_possible_actions(), [(STOP_RELATION, "B")])
        legacy = Environment(graph).start_new_query(State(None, "B", "B"), {"B"})
        self.assertEqual(legacy.get_possible_actions(), [])

    def test_horizon_termination_has_zero_potential(self):
        import networkx as nx
        from Environment import Environment, State

        graph = nx.MultiDiGraph()
        graph.add_edge("A", "B", relation="family")
        environment = Environment(graph, allow_stop=True).start_new_query(
            State(None, "A", "A"), {"B"})
        state, _ = environment.transit(
            ("family", "B"), 1, {}, {1: tf.zeros(50), 2: tf.ones(50)},
            end_of_horizon=True)
        self.assertTrue(state.terminal)
        self.assertEqual(state.h_t, {("A", "family", "B")})
        self.assertEqual(environment.get_possible_actions(), [])

    @staticmethod
    def make_policy(max_hops=3, beam_size=1):
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
        graph.add_edge("A", "B", relation="forward")
        graph.add_edge("B", "C", relation="forward")
        return TheseusSRN(graph, FakeEmbedder(), max_hops=max_hops,
                          beam_size=beam_size)

    def test_stop_vector_trainable_and_receives_gradient(self):
        from Environment import STOP_RELATION

        policy = self.make_policy()
        self.assertTrue(policy.stop_relation.trainable)
        stop_tensor = policy.relation_vector(STOP_RELATION)
        self.assertIsInstance(stop_tensor, tf.Tensor)
        self.assertEqual(stop_tensor.dtype, tf.float32)
        # Also exercise initialization, which previously failed in tf.stack.
        policy._initialize_optimizer()
        with tf.GradientTape() as tape:
            scores = policy.action_scores(
                tf.ones((4, 50)), tf.zeros(50),
                [("forward", "B"), (STOP_RELATION, "A")])
            objective = tf.reduce_sum(scores)
        gradient = tape.gradient(objective, policy.stop_relation)
        self.assertIsNotNone(gradient)
        self.assertEqual(tuple(gradient.shape), (50,))

    def test_pruning_retains_stop(self):
        from Environment import STOP_RELATION

        policy = self.make_policy()
        actions = [("r1", "B"), ("r2", "C"), ("r3", "D"),
                   (STOP_RELATION, "A")]
        kept, _ = policy.prune(actions, tf.constant([9.0, 8.0, 7.0, -40.0]), 2)
        self.assertEqual(len(kept), 2)
        self.assertIn((STOP_RELATION, "A"), kept)
        self.assertIn(("r1", "B"), kept)

    def test_learned_stop_never_becomes_evidence_triple(self):
        from Environment import STOP_RELATION

        policy = self.make_policy(max_hops=3)
        policy.action_scores = lambda q, h, actions: tf.constant([
            20.0 if relation == STOP_RELATION else -20.0
            for relation, _ in actions], tf.float32)
        record = (["who", "is", "mother"], "A", {"A"}, 3, "q1")
        self.assertEqual(policy.predict_query(record), ("A", []))
        score, predictions = policy.evaluate([record])
        self.assertEqual(score, 1.0)
        self.assertEqual(predictions[0]["stop_reason"], "learned_stop")
        self.assertEqual(predictions[0]["predicted_hops"], 0)

    def test_gold_hops_cannot_change_rollout(self):
        from Environment import STOP_RELATION

        policy = self.make_policy(max_hops=3)
        policy.action_scores = lambda q, h, actions: tf.constant([
            10.0 if relation != STOP_RELATION else -10.0
            for relation, _ in actions], tf.float32)
        record_one = (["who", "is", "mother"], "A", {"C"}, 1, "q1")
        record_three = (["who", "is", "mother"], "A", {"C"}, 3, "q2")
        first = policy.predict_query(record_one)
        second = policy.predict_query(record_three)
        self.assertEqual(first, second)
        self.assertEqual(first[0], "C")
        self.assertEqual(first[1], [
            ("A", "forward", "B"), ("B", "forward", "C")])

    def test_optimizer_prebuilds_all_hops_and_history_before_first_update(self):
        """Keras 3 must accept weights used only on later, longer episodes."""
        from Environment import STOP_RELATION

        policy = self.make_policy(max_hops=3)
        original_scores = policy.action_scores
        preference = {"stop": True}

        def biased_scores(question, history, actions):
            scores = original_scores(question, history, actions)
            bias = tf.constant([
                20.0 if (relation == STOP_RELATION) == preference["stop"]
                else -20.0
                for relation, _ in actions
            ], tf.float32)
            return scores + bias

        policy.action_scores = biased_scores
        # First update ends at source via STOP.
        short = (["who", "is", "mother"], "A", {"A"}, 1, "short")
        policy.train_query(short)
        self.assertTrue(policy._optimizer_initialized)
        initial_weights = len(policy.trainable_variables)

        # Second update uses both KG edges and additional hop-specific SLP
        # weights, which used to trigger Keras's "Unknown variable" error.
        preference["stop"] = False
        long = (["who", "is", "mother"], "A", {"C"}, 3, "long")
        prediction, loss, _ = policy.train_query(long)
        self.assertEqual(prediction, "C")
        self.assertIsNotNone(loss)
        self.assertEqual(len(policy.trainable_variables), initial_weights)

    def test_mixed_hop_cli_does_not_require_oracle(self):
        from theseus_experiment import argument_parser

        args = argument_parser().parse_args([
            "--dataset", "kinship", "--hops", "n", "--max-hops", "3"])
        self.assertEqual(args.hops, "n")
        self.assertEqual(args.max_hops, 3)
        self.assertFalse(args.oracle_hops)


if __name__ == "__main__":
    unittest.main()
