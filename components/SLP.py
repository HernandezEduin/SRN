import tensorflow as tf
from tensorflow.keras import layers
from Environment import d

initializer = tf.keras.initializers.GlorotNormal()

class SLP(layers.Layer):
    def __init__(self, T):
        super(SLP, self).__init__()
        for t in range(1, T+1):
            weight = 'weights_{}'.format(t)
            bias = 'biases_{}'.format(t)
            w = self.add_weight(name=weight, shape=(d, d), initializer="glorot_normal")
            b = self.add_weight(name=bias, shape=(d,), initializer="glorot_normal")
            setattr(self, weight, w)
            setattr(self, bias, b)
    
    def call(self, q, t):
        w_t = getattr(self, 'weights_{}'.format(t))
        b_t = getattr(self, 'biases_{}'.format(t))
        
        logits = tf.matmul(q, w_t)
        logits = logits + b_t
        q_t = tf.math.tanh(logits)

        return q_t


