"""Directed KG navigation and paper-aligned potential-based reward shaping."""

import tensorflow as tf

d = 50


class State:
    def __init__(self, q, e_s, e_t, h_t=None, t=1, q_t=None, H_t=None):
        self.q, self.e_s, self.e_t = q, e_s, e_t
        self.h_t = set() if h_t is None else h_t
        self.memory = {"t": t, "q_t": {} if q_t is None else q_t,
                       "H_t": {} if H_t is None else H_t}

    def set_memory(self, t, q_t, H_t):
        self.memory = {"t": t, "q_t": q_t, "H_t": H_t}
        return self

    def get_memory(self):
        return self.memory


class Rewards:
    def __init__(self, gamma=0.95):
        self.gamma = gamma

    def phi(self, state):
        memory = state.get_memory()
        t, questions, histories = memory["t"], memory["q_t"], memory["H_t"]
        if t <= 1:
            return tf.constant(0.0, tf.float32)
        available = [tf.reduce_sum(questions[i], axis=0) for i in range(1, t) if i in questions]
        if not available or t not in histories:
            return tf.constant(0.0, tf.float32)
        question = tf.math.l2_normalize(tf.add_n(available), axis=0)
        history = tf.math.l2_normalize(histories[t], axis=0)
        return tf.maximum(tf.reduce_sum(question * history), 0.0)

    @staticmethod
    def terminal_base_reward(final_entity, answers):
        """Published SRN base reward for every action of a completed episode."""
        return float(final_entity in answers)

    def get_transition_reward(self, cur_state, action, next_state, answers):
        """Return shaping only, since final-answer correctness is not yet known.

        SRN assigns the same base reward to every decision after the complete
        trajectory is observed. Awarding it on intermediate visits would reward
        episodes that pass through an answer but finish at the wrong entity.
        """
        shaped = self.gamma * self.phi(next_state) - self.phi(cur_state)
        return tf.stop_gradient(shaped)


class Environment:
    def __init__(self, knowledge_graph, start_state=None, rewards_func=None):
        self.knowledge_graph = knowledge_graph
        self.start_state = self.current_state = start_state
        self.rewards_func = rewards_func if rewards_func is not None else Rewards()
        self.ans = frozenset()

    def start_new_query(self, state, ans):
        self.start_state = self.current_state = state
        self.ans = frozenset(ans) if isinstance(ans, (set, frozenset, list, tuple)) else frozenset((ans,))
        return self

    def get_possible_actions(self):
        node = self.current_state.e_t
        if not self.knowledge_graph.has_node(node):
            return []
        if self.knowledge_graph.is_multigraph():
            return [(data["relation"], target)
                    for _, target, data in self.knowledge_graph.out_edges(node, data=True)]
        return [(data["relation"], target)
                for _, target, data in self.knowledge_graph.out_edges(node, data=True)]

    def get_next_state(self, action):
        relation, neighbor = action
        if action not in self.get_possible_actions():
            raise ValueError("Invalid directed relation/target transition: {}".format(action))
        state = self.current_state
        return State(state.q, state.e_s, neighbor, state.h_t | {(state.e_t, relation, neighbor)})

    def transit(self, action, t, q_t, H_t):
        next_state = self.get_next_state(action).set_memory(t + 1, q_t, H_t)
        reward = self.rewards_func.get_transition_reward(self.current_state, action, next_state, self.ans)
        self.current_state = next_state
        return next_state, reward
