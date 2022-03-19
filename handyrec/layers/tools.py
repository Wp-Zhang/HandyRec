import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Zeros
from typing import List


class SequencePoolingLayer(Layer):
    """Pooling layer for sequence feature"""

    def __init__(self, method: str, **kwargs):
        super().__init__(**kwargs)

        assert method in [
            "mean",
            "max",
            "sum",
        ], "Pooling method should be `mean`, `max`, or `sum`"
        self.method = method
        # self.eps = tf.constant(1e-8, tf.float32)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, mask):
        if mask is None:
            raise ValueError("Embedding layer should set `mask_zero` as True")
        # * inputs: (batch, seq_max_len, emb_dim)
        # * mask: (batch, seq_max_len)
        # * output: (batch, 1, emb_dim)

        mask = tf.dtypes.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)  # (batch, seq_max_len, 1)

        if self.method == "max":
            output = inputs - (1 - mask) * 1e9
            return tf.reduce_max(output, axis=1, keepdims=True)

        elif self.method == "sum":
            output = inputs * mask
            return tf.reduce_sum(output, axis=1, keepdims=True)

        elif self.method == "mean":
            mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
            mask_weight = tf.math.divide_no_nan(mask, mask_sum)
            output = inputs * mask_weight
            return tf.reduce_sum(output, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[-1])

    def get_config(self):
        config = {"method": self.method}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EmbeddingIndex(Layer):
    """Output full index of embedding nomatter input"""

    def __init__(self, index: List[int], **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self):
        config = {"index": self.index}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SampledSoftmaxLayer(Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        self.size = None
        self.zero_bias = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(
            shape=[self.size],
            initializer=Zeros,
            dtype=tf.float32,
            trainable=False,
            name="bias",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        `inputs` is a tuple with length as 3.
            inputs[0] is embedding of all items
            inputs[1] is user embedding
            inputs[2] is the index of label items for each user
        """
        item_embeddings, user_embeddings, item_idx = inputs

        loss = tf.nn.sampled_softmax_loss(
            weights=item_embeddings,
            biases=self.zero_bias,
            labels=item_idx,
            inputs=user_embeddings,
            num_sampled=self.num_sampled,
            num_classes=self.size,
        )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = {"num_sampled": self.num_sampled}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
