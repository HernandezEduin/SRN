"""Executable SRN policy for THESEUS data.

Preserves BiGRU, hop-specific SLP, relation attention, semantic perceptron,
GRU history, answer-only potential shaping, and on-policy REINFORCE.
Reference paths are never supplied to the policy or loss.
"""

import numpy as np
import tensorflow as tf

from Environment import Environment, Rewards, State, STOP_RELATION, d
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
        self.env = Environment(graph, rewards_func=Rewards(gamma=0.95), allow_stop=True)
        # The paper specifies a learnable STOP action but not its initialization.
        # Use a separate trainable relation-sized vector. TransE files remain unchanged.
        self.stop_relation = self.add_weight(
            name="stop_relation", shape=(d,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
        )
        self.BiGRU = BiGRU()
        self.SLP = SLP(max_hops)
        self.Attention = Attention()
        self.Perceptron = Perceptron()
        self.GRU = GRU()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def encode(self, words, training=False):
        vectors = np.stack([self.Embedder.embed_word(word) for word in words])
        return self.BiGRU(tf.convert_to_tensor(vectors[None, :, :], tf.float32), training=training)

    def relation_vector(self, relation):
        if relation == STOP_RELATION:
            return self.stop_relation
        vector = self.Embedder.embed_relation(relation)
        if vector is None:
            raise ValueError("Missing frozen TransE vector for relation {}".format(relation))
        return tf.convert_to_tensor(vector, tf.float32)

    def action_scores(self, question_at_t, history, actions):
        """Score each KG edge using a batched computation over unique relations.

        The original SRN score is relation-only. Duplicate relations receive
        identical scores, and gradients still flow through attention and the
        semantic perceptron exactly as in their scalar implementations.
        """
        relation_ids = {}
        vectors = []
        indices = []
        for relation, _ in actions:
            if relation not in relation_ids:
                relation_ids[relation] = len(vectors)
                vectors.append(self.relation_vector(relation))
            indices.append(relation_ids[relation])
        if not vectors:
            return tf.zeros((0,), tf.float32)
        # tf.stack preserves the gradient to the trainable STOP vector.
        relation_vectors = tf.stack(vectors)
        attended = self.Attention.call_many(relation_vectors, question_at_t)
        unique_scores = self.Perceptron.call_many(relation_vectors, history, attended)
        return tf.gather(unique_scores, indices)

    def prune(self, actions, logits, count):
        if len(actions) <= count:
            return actions, logits
        stop_indices = [i for i, (relation, _) in enumerate(actions)
                        if relation == STOP_RELATION]
        if stop_indices:
            # STOP must remain available even at high-degree KG nodes.
            if count < 2:
                raise ValueError("max_actions must be at least 2 when STOP is enabled")
            stop_index = stop_indices[0]
            other_indices = [i for i in range(len(actions)) if i != stop_index]
            top = tf.math.top_k(tf.gather(logits, other_indices),
                                k=min(count - 1, len(other_indices))).indices.numpy()
            indices = [other_indices[int(i)] for i in top] + [stop_index]
        else:
            indices = tf.math.top_k(logits, k=count).indices.numpy().tolist()
        return [actions[index] for index in indices], tf.gather(logits, indices)

    def train_query(self, record):
        words, start, answers, _, _ = record
        # Gold hop annotations must never determine the number of rollout steps.
        with tf.GradientTape() as tape:
            encoded = self.encode(words, training=True)
            self.env.start_new_query(State(encoded, start, start), answers)
            history = tf.zeros(d, tf.float32)
            gru_states = None
            question_mem, history_mem = {}, {1: history}
            log_probs, rewards = [], []
            for t in range(1, self.max_hops + 1):
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
                if selected[0] != STOP_RELATION and t < self.max_hops:
                    history, gru_states = self.GRU(
                        self.relation_vector(selected[0]), states=gru_states,
                        training=True)
                history_mem[t + 1] = history
                next_state, shaping_reward = self.env.transit(
                    selected, t, question_mem, history_mem,
                    end_of_horizon=(t == self.max_hops))
                rewards.append(shaping_reward)
                if next_state.terminal:
                    break
            prediction = self.env.current_state.e_t
            if not log_probs:
                return prediction, None, []
            # SRN paper, Sec. 2.4: all decisions in a completed successful
            # trajectory receive base R=1, even when earlier nodes are not
            # answers. Intermediate hits in unsuccessful episodes receive R=0.
            terminal_reward = tf.constant(
                self.env.rewards_func.terminal_base_reward(prediction, answers),
                tf.float32
            )
            rewards = [tf.stop_gradient(shaping + terminal_reward)
                       for shaping in rewards]
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

    def _predict_query_details(self, record):
        """Beam search with absorbing STOP actions and no gold-hop oracle.

        Completed branches retain their score at later steps. A self-loop
        STOP is not appended as a KG evidence edge to the returned path.
        GRU updates are computed only for surviving, nonterminal branches.
        """
        words, start, _, _, _ = record
        encoded = self.encode(words, training=False)
        step_questions = {
            t: self.SLP(encoded, t=t)
            for t in range(1, self.max_hops + 1)
        }
        # node, log probability, GRU history, states, evidence, done, reason
        beams = [(start, 0.0, tf.zeros(d, tf.float32), None, [], False, None)]
        for t in range(1, self.max_hops + 1):
            candidates = []
            for node, path_score, history, states, path, done, reason in beams:
                if done:
                    candidates.append((node, path_score, history, states, path,
                                       True, reason, None))
                    continue
                self.env.start_new_query(State(encoded, start, node), frozenset())
                actions = self.env.get_possible_actions()
                if not actions:
                    candidates.append((node, path_score, history, states, path,
                                       True, "dead_end", None))
                    continue
                logits = self.action_scores(step_questions[t], history, actions)
                actions, logits = self.prune(actions, logits, self.max_actions)
                log_probabilities = tf.nn.log_softmax(logits).numpy()
                for index in np.argsort(-log_probabilities)[:self.beam_size]:
                    relation, target = actions[int(index)]
                    stopped = relation == STOP_RELATION
                    done_now = stopped or t == self.max_hops
                    next_path = (path if stopped else
                                 path + [(node, relation, target)])
                    candidates.append((
                        target, path_score + float(log_probabilities[index]),
                        history, states, next_path, done_now,
                        "learned_stop" if stopped else
                        "max_horizon" if done_now else None,
                        relation if not done_now else None,
                    ))
            best = sorted(candidates, key=lambda branch: branch[1],
                          reverse=True)[:self.beam_size]
            if not best:
                break
            beams = []
            for target, score, history, states, path, done, reason, relation in best:
                if relation is not None:
                    next_history, next_states = self.GRU(
                        self.relation_vector(relation), states=states,
                        training=False)
                    beams.append((target, score, next_history, next_states,
                                  path, done, reason))
                else:
                    beams.append((target, score, history, states,
                                  path, done, reason))
            if all(branch[5] for branch in beams):
                break
        if not beams:
            return start, [], "dead_end"
        winner = beams[0]
        return winner[0], winner[4], winner[6]

    def predict_query(self, record):
        prediction, path, _ = self._predict_query_details(record)
        return prediction, path

    def evaluate(self, examples):
        predictions, correct = [], 0
        for record in examples:
            prediction, path, stop_reason = self._predict_query_details(record)
            hit = prediction in record[2]
            correct += int(hit)
            predictions.append({"id": record[4], "hops": record[3],
                                "start": record[1], "prediction": prediction,
                                "correct": hit, "path": path,
                                "predicted_hops": len(path),
                                "stop_reason": stop_reason})
        return correct / len(examples) if examples else 0.0, predictions
