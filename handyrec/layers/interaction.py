import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.initializers import Zeros


class FM(Layer):
    """Factorization Machine"""

    def __init__(self, **kwargs):
        self.linear = None
        self.w_0 = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.linear = Dense(1, use_bias=False)
        self.w_0 = self.add_weight(
            shape=(1,),
            initializer=Zeros,
            dtype=tf.float32,
            trainable=True,
            name="W_0",
        )

    def call(self, inputs, mask=None, *args, **kwargs):
        # * inputs: (batch_size, num_of_fields, embedding_dim)
        # * part2: (batch_size, 1)
        part2 = tf.reduce_sum(self.linear(inputs), axis=1, keepdims=False)

        # * square_sum: (batch_size, embedding_dim)
        # * sum_square: (batch_size, embedding_dim)
        square_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=False))
        sum_square = tf.reduce_sum(inputs * inputs, axis=1, keepdims=False)

        # * part3: (batch_size, 1)
        part3 = square_sum - sum_square
        part3 = 0.5 * tf.reduce_sum(part3, axis=1, keepdims=True)
        return tf.nn.bias_add(part2 + part3, self.w_0)

    def compute_output_shape(self, input_shape):
        return (None, 1)
