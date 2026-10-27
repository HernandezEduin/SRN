import tensorflow as tf
from tensorflow.keras import layers
from Environment import d

initializer = tf.keras.initializers.GlorotNormal()

class Attention(layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        self.W = self.add_weight(name="attention_weight", shape=(d,), initializer="glorot_normal")
        self.b = self.add_weight(name="attention_bias", shape=(), initializer="zeros")

    def call(self, r_star, q_t):
        if r_star.shape == (d, 1):
            r_star = tf.transpose(r_star)

        r_star = tf.reshape(r_star, [d])

        beta_stars = tf.reduce_sum(q_t * r_star[None, :] * self.W[None, :], axis=-1) + self.b
        alpha_stars = tf.nn.softmax(beta_stars, axis=0)
        
        q_t_stars = tf.math.multiply(alpha_stars[:, None], q_t)
        q_t_star = tf.reduce_sum(q_t_stars, 0)

        return q_t_star

    def beta_stars(self, r_star, w_t_m):
        output = r_star * w_t_m
        b_star = self.W * output + self.b
        return b_star
