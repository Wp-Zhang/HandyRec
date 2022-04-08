"""Contains some core layers may be used by different models.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.regularizers import l2
from typing import Tuple
from .utils import get_activation_layer


class DNN(Layer):
    """DNN layer"""

    def __init__(
        self,
        hidden_units: Tuple[int],
        activation: str = "relu",
        l2_reg: float = 0,
        dropout_rate: float = 0,
        use_bn: bool = False,
        output_activation: str = None,
        seed: int = 2022,
        **kwargs
    ):
        """
        Parameters
        ----------
        hidden_units : Tuple[int]
            DNN structure.
        activation : str, optional
            Activation function for each layers except the last one, by default ``"relu"``.
        l2_reg : float, optional
            L2 regularization param, by default ``0``.
        dropout_rate : float, optional
            Dropout rate, by default ``0``.
        use_bn : bool, optional
            Whether use batch normalization or not, by default ``False``.
        output_activation : str, optional
            Activation function for the last layer, by default ``None``.
        seed : int, optional
            Random seef for dropout, by default ``2022``.
        """
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed
        self.layers = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)

        self.layers = []
        for i, unit in enumerate(hidden_units):
            dense_layer = Dense(
                unit,
                kernel_regularizer=l2(self.l2_reg),
            )
            self.layers.append(dense_layer)
            if i + 1 != len(hidden_units) and self.activation:
                self.layers.append(get_activation_layer(self.activation))
            elif self.output_activation:
                self.layers.append(get_activation_layer(self.output_activation))

            if self.use_bn:
                self.layers.append(BatchNormalization())
            self.layers.append(Dropout(self.dropout_rate, seed=self.seed + i))

        self.layers = Sequential(self.layers)

    def call(self, inputs, **kwargs):
        return self.layers(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self):
        config = {
            "hidden_units": self.hidden_units,
            "activation": self.activation,
            "l2_reg": self.l2_reg,
            "dropout_rate": self.dropout_rate,
            "use_bn": self.use_bn,
            "output_activation": self.output_activation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**config, **base_config}


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
