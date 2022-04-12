"""Contains some core layers may be used by different models.
"""
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout
from tensorflow.keras import Sequential
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
