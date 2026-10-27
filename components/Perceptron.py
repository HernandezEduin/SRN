import tensorflow as tf
from tensorflow.keras import layers
from Environment import d

initializer = tf.keras.initializers.GlorotNormal()

class Perceptron(layers.Layer):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.W_L1 = self.add_weight(name="semantic_w1", shape=(d, 2*d), initializer="glorot_normal")
        self.W_L2 = self.add_weight(name="semantic_w2", shape=(d,), initializer="glorot_normal")

    def call(self, r_star, H_t, q_t_star):
        if r_star.shape == (d, 1):
            r_star = tf.transpose(r_star)

        r_star = tf.reshape(r_star, [d])

        concat_input = tf.reshape(tf.concat([H_t, q_t_star], 0), [2*d, 1])
        
        logits_1 = tf.reshape(tf.matmul(self.W_L1, concat_input), [d])
        out_1 = tf.nn.relu(logits_1)
        out_2 = tf.math.multiply(self.W_L2, out_1)
        semantic_score = tf.math.multiply(r_star, out_2)
        semantic_score = tf.reduce_sum(semantic_score, 0)

        return semantic_score

    def call_many(self, relation_vectors, history, attended_questions):
        """Vectorized equivalent of call() with one shared history."""
        histories = tf.broadcast_to(history[None, :], tf.shape(relation_vectors))
        features = tf.concat([histories, attended_questions], axis=1)
        hidden = tf.nn.relu(tf.matmul(features, self.W_L1, transpose_b=True))
        return tf.reduce_sum(
            relation_vectors * hidden * self.W_L2[None, :], axis=1
        )

