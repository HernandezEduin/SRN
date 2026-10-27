import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Environment import d

HIDDEN_DIM = int(d / 2)
dropout = 0.3

class BiGRU(layers.Layer):
    def __init__(self):
        super(BiGRU, self).__init__()
        self.fw_1 = keras.layers.GRU(HIDDEN_DIM, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
        self.fw_2 = keras.layers.GRU(HIDDEN_DIM, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
        self.bw_1 = keras.layers.GRU(HIDDEN_DIM, go_backwards=True, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
        self.bw_2 = keras.layers.GRU(HIDDEN_DIM, go_backwards=True, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)

    def call(self, q, training=False):
        n = q.shape[1]

        fw_out_1 = self.fw_1(q, training=training)
        bw_out_1 = self.bw_1(q, training=training)
        hidden_out = tf.concat([fw_out_1, tf.reverse(bw_out_1, axis=[1])], axis=-1)
        
        fw_out_2 = self.fw_2(hidden_out, training=training)
        bw_out_2 = self.bw_2(hidden_out, training=training)
        output = tf.concat([fw_out_2, tf.reverse(bw_out_2, axis=[1])], axis=-1)
        output = tf.reshape(output, (n, d))

        return output

