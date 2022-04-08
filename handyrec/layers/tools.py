"""Contains some tool layers.

"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
from tensorflow.keras.initializers import Zeros
from typing import List
from .layers import DNN


class SequencePoolingLayer(Layer):
    """Pooling layer for sequence feature"""

    def __init__(self, method: str, **kwargs):
        """
        Parameters
        ----------
        method : str
            Pooling method.
        """
        super().__init__(**kwargs)

        assert method in [
            "mean",
            "max",
            "sum",
        ], "Pooling method should be `mean`, `max`, or `sum`"
        self.method = method

    def call(self, inputs, mask=None):
        if mask is None:
            raise ValueError("Embedding layer should set `mask_zero` as True")
        # * inputs: (batch, seq_max_len, emb_dim)
        # * mask: (batch, seq_max_len, emb_dim)
        # * output: (batch, 1, emb_dim)
        mask = tf.dtypes.cast(mask, tf.float32)

        if self.method == "max":
            output = inputs - (1 - mask) * 1e9
            return tf.reduce_max(output, axis=1, keepdims=True)

        elif self.method == "sum":
            output = inputs * mask
            return tf.reduce_sum(output, axis=1, keepdims=True)

        else:  # * self.method == "mean"
            mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
            mask_weight = tf.math.divide_no_nan(mask, mask_sum)
            output = inputs * mask_weight
            return tf.reduce_sum(output, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[-1])

    def get_config(self):
        config = {"method": self.method}
        base_config = super().get_config()
        return {**config, **base_config}


class ValueTable(Layer):
    """Output a full list of values of a feature to be the input of embedding layer"""

    def __init__(self, value_list: List, **kwargs):
        """
        Parameters
        ----------
        value_list : List
            Feature values of all items.
        """
        self.value = tf.constant(value_list)
        super().__init__(**kwargs)

    def call(self, *args, **kwargs):
        return self.value

    def get_config(self):
        return super().get_config()


class SampledSoftmaxLayer(Layer):
    """Sampled softmax"""

    def __init__(self, num_sampled=5, **kwargs):
        """Parameters
        ----------
        num_sampled : int, optional
            Number of sampled negative samples, by default ``5``.
        """
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
        Note
        ----
        `inputs` is a tuple with length of 3.
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
        return {**config, **base_config}


class CustomEmbedding(Embedding):
    """
    Rewrite official embedding layer so that masked and un-masked
        embeddings can be concatenated together.
    """

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        # * Rewrite compute_mask
        mask = tf.not_equal(inputs, 0)  # (?, n)
        mask = tf.expand_dims(mask, axis=-1)  # (?, n, 1)
        tile_shape = [1] * (len(mask.shape) - 1) + [self.output_dim]
        mask = tf.tile(mask, tile_shape)  # (?, n, output_dim)
        return mask


class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN"""

    def __init__(
        self,
        hidden_units=(64, 32, 1),
        activation="sigmoid",
        l2_reg=0,
        dropout_rate=0,
        use_bn=False,
        seed=1024,
        **kwargs
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.dnn = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.input_check(input_shape)
        self.dnn = DNN(
            self.hidden_units,
            self.activation,
            self.l2_reg,
            self.dropout_rate,
            self.use_bn,
            seed=self.seed,
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, keys = inputs  # * (?, 1, embedding_size) and (?, T, embedding_size)
        keys_len = keys.get_shape()[1]  # * T
        queries = tf.repeat(query, keys_len, 1)  # * (?, T, embedding_size)
        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        # * att_input: (?, T, embedding_size * 4), att_output: (?, T, 1)
        att_out = self.dnn(att_input)
        att_out = tf.transpose(att_out, [0, 2, 1])  # * (?, 1, T)

        return att_out

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(
        self,
    ):
        config = {
            "activation": self.activation,
            "hidden_units": self.hidden_units,
            "l2_reg": self.l2_reg,
            "dropout_rate": self.dropout_rate,
            "use_bn": self.use_bn,
            "seed": self.seed,
        }
        base_config = super(LocalActivationUnit, self).get_config()
        return {**config, **base_config}

    def input_check(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "A `LocalActivationUnit` layer should be called "
                "on a list of 2 inputs"
            )

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d and %d, expect to be 3 dimensions"
                % (len(input_shape[0]), len(input_shape[1]))
            )

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError(
                "A `LocalActivationUnit` layer requires "
                "inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)"
                "Got different shapes: %s,%s" % (input_shape[0], input_shape[1])
            )


# class Similarity(Layer):
#     def __init__(self, type: str, **kwargs):
#         self.type = type
#         super().__init__(**kwargs)

#     def call(self, inputs, *args, **kwargs):
#         embd_a, embd_b = inputs
#         if self.type == "cos":
#             embd_a = tf.nn.l2_normalize(embd_a, axis=-1)
#             embd_b = tf.nn.l2_normalize(embd_b, axis=-1)
#         output = tf.reduce_sum(tf.multiply(embd_a, embd_b), axis=-1, keepdims=True)
#         return output

#     def get_config(self):
#         config = {"type": self.type}
#         base_config = super().get_config()
#         return {**config, **base_config}
