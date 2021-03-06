from typing import OrderedDict
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, LayerNormalization, Dropout, Input
from handyrec.features import FeatureGroup
from handyrec.layers import DNN, PositionEmbedding
from handyrec.layers.utils import concat


class FilterLayer(Layer):
    """Filter layer in FMLP-Rec

    References
    ----------
    .. [1] Source code of FMLP-Rec, https://github.com/Woeee/FMLP-Rec
    """

    def __init__(self, dropout: float = 0, layer_norm_eps: float = 1e-12, **kwargs):
        self.weight = None
        self.weight_complex = None
        self.output_dropout = Dropout(dropout)
        self.layernorm = LayerNormalization(epsilon=layer_norm_eps)
        super().__init__(**kwargs)

    def build(self, input_shape):
        _, seq_len, hidden_size = input_shape
        self.weight = self.add_weight("real_w", (1, hidden_size, seq_len // 2 + 1))
        self.weight_complex = self.add_weight(
            "complex_w", (1, hidden_size, seq_len // 2 + 1)
        )

        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        weight = tf.complex(self.weight, self.weight_complex)
        # * rfft in TF is fixed to apply on the last axis, so we need to
        # * transpose the input
        embd_fft = tf.signal.rfft(tf.transpose(inputs, perm=[0, 2, 1]))
        embd_fft = embd_fft * weight
        embd_fft = tf.signal.irfft(embd_fft)
        # * Note: if seq_len is odd, the last dimension of embd_fft will
        # * be seq_len-1, so seq_len needs to be even.
        embd_fft = tf.transpose(embd_fft, perm=[0, 2, 1])  # * transpose back
        embd_fft = self.output_dropout(embd_fft)
        output = self.layernorm(embd_fft + inputs)

        return output


class FilterBlock(Layer):
    """Filter Block in FMLP-Rec

    References
    ----------
    .. [1] Source code of FMLP-Rec, https://github.com/Woeee/FMLP-Rec
    """

    def __init__(
        self,
        dnn_activation: str = "gelu",
        dropout: float = 0,
        filter_dropout: float = 0,
        layer_norm_eps: float = 1e-12,
        **kwargs,
    ):
        self.activation = dnn_activation
        self.filter_layer = FilterLayer(filter_dropout, layer_norm_eps)
        self.dnn_layer = None
        self.dropout = Dropout(dropout)
        self.layernorm = LayerNormalization(epsilon=layer_norm_eps)
        super().__init__(**kwargs)

    def build(self, input_shape):
        _, _, hidden_size = input_shape
        dnn_unit = (hidden_size * 4, hidden_size)
        self.dnn_layer = DNN(dnn_unit, self.activation)

        return super().build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        filtered = self.filter_layer(inputs)
        output = self.dnn_layer(filtered)
        output = self.layernorm(self.dropout(output) + inputs)
        if mask is not None:
            output = output * tf.cast(mask, tf.float32)
        return output


def FMLPRec(
    seq_feat_group: FeatureGroup,
    dropout: float = 0,
    block_num: int = 1,
    layer_norm_eps: float = 1e-12,
) -> Model:
    """Implementation of FMLP-Rec Model

    Parameters
    ----------
    seq_feat_group : FeatureGroup
        Feature group of item sequence features.
    dropout : float, optional
        Dropout rate of concatenated embeddings, by default ``0``.
    block_num : int, optional
        Number of filter blocks, by default ``1``.
    layer_norm_eps : float, optional
        Epsilon of layer normalization, by default ``1e-12``.

    Returns
    -------
    Model
        A FMLP-Rec model.

    References
    ----------
    .. [1] Kun Zhou, Hui Yu, Wayne Xin Zhao, Ji-Rong Wen. "Filter-enhanced MLP is All You
        Need for Sequential Recommendation." The 31st conference in the International World
        Wide Web Conference. 2022.
    """

    embd_outputs = OrderedDict()
    pos_embd_outputs = OrderedDict()
    neg_embd_outputs = OrderedDict()
    pos_item_input = None
    neg_item_input = None
    for feat in seq_feat_group.features:
        if feat.seq_len % 2 == 1:
            raise AttributeError(
                f"`seq_len` of feature {feat.name} is odd, needs to be even"
            )
        sparse_embd = seq_feat_group.embd_layers[feat.unit.name]
        seq_input = seq_feat_group.input_layers[feat.name]
        embd_seq = sparse_embd(seq_input)  # * (batch_size, seq_len, embd_dim)

        # * Get embeddings of positive and negative target items
        if pos_item_input is None:
            pos_item_input = Input((1,), name=feat.unit.name, dtype=tf.int32)
            neg_item_input = Input((1,), name="neg_" + feat.unit.name, dtype=tf.int32)

        embd_pos = sparse_embd(pos_item_input)
        mask = sparse_embd.compute_mask(pos_item_input)
        embd_neg = sparse_embd(neg_item_input)
        pos_embd_outputs[feat.name] = embd_pos
        neg_embd_outputs[feat.name] = embd_neg

        # * Embedding Part (batch_size, seq_len, embd_dim)
        position_embd = PositionEmbedding()
        embd_seq = embd_seq + position_embd(embd_seq)
        embd_seq = LayerNormalization(epsilon=layer_norm_eps)(embd_seq)
        embd_seq = Dropout(dropout)(embd_seq)

        # * Learnable Filter-enhanced Blocks
        for _ in range(block_num):
            embd_seq = FilterBlock(
                dnn_activation="relu",
                dropout=dropout,
                filter_dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )(embd_seq, mask=mask)

        # * (batch_size, embd_dim)
        embd_outputs[feat.name] = embd_seq[:, -1, :]

    # * Calculate loss
    output = concat([], list(embd_outputs.values()))
    pos_embd = concat([], list(pos_embd_outputs.values()))
    neg_embd = concat([], list(neg_embd_outputs.values()))

    pos_logits = tf.reduce_sum(output * pos_embd, axis=-1)
    neg_logits = tf.reduce_sum(output * neg_embd, axis=-1)
    pos_p = tf.clip_by_value(tf.sigmoid(pos_logits), 1e-8, 1 - 1e-8)
    neg_p = tf.clip_by_value(tf.sigmoid(neg_logits), 1e-8, 1 - 1e-8)
    loss = tf.reduce_mean(-tf.math.log(pos_p) - tf.math.log(1 - neg_p))

    # * Construct model
    seq_feat_inputs = list(seq_feat_group.input_layers.values())
    inputs = [pos_item_input, neg_item_input] + seq_feat_inputs
    model = Model(inputs=inputs, outputs=loss)
    model.add_loss(loss)

    model.__setattr__("real_inputs", seq_feat_inputs + [pos_item_input])
    model.__setattr__("real_outputs", pos_logits)

    return model
