import tensorflow as tf
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.layers import Layer


class Dice(Layer):
    """The Data Adaptive Activation Function in DIN

    References
    ----------
    .. [1] Weichen Shen. (2017). DeepCTR: Easy-to-use, Modular and Extendible package of
        deep-learning based CTR models. https://github.com/shenweichen/deepctr.
    """

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bn = None
        self.alphas = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False
        )
        self.alphas = self.add_weight(
            shape=(input_shape[-1],),
            initializer=Zeros(),
            dtype=tf.float32,
            name="dice_alpha",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs_normed = self.bn(inputs, training=kwargs.get("training", False))
        x_p = tf.sigmoid(inputs_normed)
        return x_p * inputs + (1.0 - x_p) * self.alphas * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(
        self,
    ):
        config = {"axis": self.axis, "epsilon": self.epsilon}
        base_config = super().get_config()
        return {**config, **base_config}
