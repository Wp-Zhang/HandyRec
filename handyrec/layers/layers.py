import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from typing import Tuple


class DNN(Layer):
    """DNN"""

    def __init__(
        self,
        hidden_units: Tuple[int],
        activation="relu",
        l2_reg=0,
        dropout_rate=0,
        use_bn=False,
        output_activation=None,
        seed=1024,
        **kwargs
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)

        self.layers = []
        for i, unit in enumerate(hidden_units):
            if i + 1 != len(hidden_units):
                dense_layer = Dense(
                    unit,
                    activation=self.activation,
                    kernel_regularizer=l2(self.l2_reg),
                )
            else:
                dense_layer = Dense(
                    unit,
                    activation=self.output_activation,
                    kernel_regularizer=l2(self.l2_reg),
                )
            self.layers += [
                dense_layer,
                BatchNormalization(),
                Dropout(self.dropout_rate, seed=self.seed + i),
            ]
        self.layers = Sequential(self.layers)

    def call(self, inputs, training=None, **kwargs):
        return self.layers(inputs, training)

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)
