"""Executable SRN policy for THESEUS data.

Preserves BiGRU, hop-specific SLP, relation attention, semantic perceptron,
GRU history, answer-only potential shaping, and on-policy REINFORCE.
Reference paths are never supplied to the policy or loss.
"""

import numpy as np
import tensorflow as tf

from Environment import Environment, Rewards, State, d
from components import Attention, BiGRU, GRU, Perceptron, SLP


class TheseusSRN(tf.keras.Model):
    def __init__(self, graph, embedder, max_hops, learning_rate=1e-3,
                 beam_size=32, max_actions=200, discount=0.9):
        super().__init__()
        self.graph = graph
        self.Embedder = embedder
        self.max_hops = max_hops
        self.beam_size = beam_size
        self.max_actions = max_actions
        self.discount = discount
        self.env = Environment(graph, rewards_func=Rewards(gamma=0.95))
        self.BiGRU = BiGRU()
        self.SLP = SLP(max_hops)
        self.Attention = Attention()
        self.Perceptron = Perceptron()
        self.GRU = GRU()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def encode(self, words, training=False):
        vectors = np.stack([self.Embedder.embed_word(word) for word in words])
        return self.BiGRU(tf.convert_to_tensor(vectors[None, :, :], tf.float32), training=training)

    def action_scores(self, question_at_t, history, actions):
        # SRN scores the relation of each KG action. Different neighbors with
        # the same relation therefore receive the same relation-based score.
        unique = {}
        scores = []
        for relation, _ in actions:
            if relation not in unique:
                vector = self.Embedder.embed_relation(relation)
                if vector is None:
                    raise ValueError("Missing frozen TransE vector for relation {}".format(relation))
                relation_vector = tf.convert_to_tensor(vector, tf.float32)
                attended = self.Attention(relation_vector, question_at_t)
                unique[relation] = self.Perceptron(relation_vector, history, attended)
            scores.append(unique[relation])
        return tf.stack(scores)

    def prune(self, actions, logits, count):
        if len(actions) <= count:
            return actions, logits
        indices = tf.math.top_k(logits, k=count).indices.numpy().tolist()
        return [actions[index] for index in indices], tf.gather(logits, indices)

    def train_query(self, record):
        words, start, answers, hops, _ = record
        if hops > self.max_hops:
            raise ValueError("Question horizon exceeds max_hops")
        with tf.GradientTape() as tape:
            encoded = self.encode(words, training=True)
            self.env.start_new_query(State(encoded, start, start), answers)
            history = tf.zeros(d, tf.float32)
            gru_states = None
            question_mem, history_mem = {}, {1: history}
            log_probs, rewards = [], []
            for t in range(1, hops + 1):
                actions = self.env.get_possible_actions()
                if not actions:
                    break
                question_mem[t] = self.SLP(encoded, t=t)
                logits = self.action_scores(question_mem[t], history, actions)
                actions, logits = self.prune(actions, logits, self.max_actions)
                log_distribution = tf.nn.log_softmax(logits)
                sampled = int(tf.random.categorical(log_distribution[None, :], 1)[0, 0])
                selected = actions[sampled]
                log_probs.append(tf.gather(log_distribution, sampled))
                relation_vector = tf.convert_to_tensor(
                    self.Embedder.embed_relation(selected[0]), tf.float32
                )
                history, gru_states = self.GRU(relation_vector, states=gru_states, training=True)
                history_mem[t + 1] = history
                _, reward = self.env.transit(selected, t, question_mem, history_mem)
                rewards.append(reward)
            prediction = self.env.current_state.e_t
            if not log_probs:
                return prediction, None, []
            running = tf.constant(0.0, tf.float32)
            returns = []
            for reward in reversed(rewards):
                running = reward + self.discount * running
                returns.append(running)
            returns.reverse()
            # Reward shaping supplies a baseline-like signal. Only selected
            # action log probabilities receive policy-gradient derivatives.
            loss = -tf.reduce_sum(tf.stack(log_probs) * tf.stop_gradient(tf.stack(returns)))
        grads = tape.gradient(loss, self.trainable_variables)
        pairs = [(grad, variable) for grad, variable in zip(grads, self.trainable_variables) if grad is not None]
        if pairs:
            self.optimizer.apply_gradients(pairs)
        return prediction, float(loss.numpy()), list(self.env.current_state.h_t)

    def predict_query(self, record):
        """Answer-blind, relation-score-based beam search."""
        words, start, _, hops, _ = record
        encoded = self.encode(words, training=False)
        step_questions = {t: self.SLP(encoded, t=t) for t in range(1, hops + 1)}
        # beam: node, log probability, history vector, GRU states, trajectory
        beams = [(start, 0.0, tf.zeros(d, tf.float32), None, [])]
        for t in range(1, hops + 1):
            candidates = []
            for node, path_score, history, states, path in beams:
                self.env.start_new_query(State(encoded, start, node), frozenset())
                actions = self.env.get_possible_actions()
                if not actions:
                    candidates.append((node, path_score, history, states, path))
                    continue
                logits = self.action_scores(step_questions[t], history, actions)
                actions, logits = self.prune(actions, logits, self.max_actions)
                log_probabilities = tf.nn.log_softmax(logits).numpy()
                best_indices = np.argsort(-log_probabilities)[:self.beam_size]
                for index in best_indices:
                    relation, target = actions[int(index)]
                    relation_vector = tf.convert_to_tensor(
                        self.Embedder.embed_relation(relation), tf.float32
                    )
                    next_history, next_states = self.GRU(relation_vector, states=states, training=False)
                    candidates.append((target, path_score + float(log_probabilities[index]),
                                       next_history, next_states, path + [(node, relation, target)]))
            beams = sorted(candidates, key=lambda branch: branch[1], reverse=True)[:self.beam_size]
            if not beams:
                break
        best = beams[0]
        return best[0], best[4]

    def evaluate(self, examples):
        predictions, correct = [], 0
        for record in examples:
            prediction, path = self.predict_query(record)
            hit = prediction in record[2]
            correct += int(hit)
            predictions.append({"id": record[4], "hops": record[3],
                                "start": record[1], "prediction": prediction,
                                "correct": hit, "path": path})
        return correct / len(examples) if examples else 0.0, predictions
