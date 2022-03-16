import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout
from tensorflow.keras import Input, Sequential
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.regularizers import l2
from typing import Tuple, List


class SequencePoolingLayer(Layer):
    """Pooling layer for sequence feature"""

    def __init__(self, method: str, **kwargs):
        super(SequencePoolingLayer, self).__init__(**kwargs)

        assert method in [
            "mean",
            "max",
            "sum",
        ], "Pooling method should be `mean`, `max`, or `sum`"
        self.method = method
        # self.eps = tf.constant(1e-8, tf.float32)

    def build(self, input_shape):
        super(SequencePoolingLayer, self).build(input_shape)

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


class EmbeddingIndex(Layer):
    """Output full index of embedding nomatter input"""

    def __init__(self, index: List[int], **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(input_shape)

    def call(self, x, **kwargs):
        return tf.constant(self.index)

    def get_config(self):
        config = {"index": self.index}
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SampledSoftmaxLayer(Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(
            shape=[self.size],
            initializer=Zeros,
            dtype=tf.float32,
            trainable=False,
            name="bias",
        )
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument
        """
        embeddings, inputs, label_idx = inputs_with_label_idx

        loss = tf.nn.sampled_softmax_loss(
            weights=embeddings,  # self.item_embedding
            biases=self.zero_bias,
            labels=label_idx,
            inputs=inputs,
            num_sampled=self.num_sampled,
            num_classes=self.size,
        )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = {"num_sampled": self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DNN, self).build(input_shape)

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
