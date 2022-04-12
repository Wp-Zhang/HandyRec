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
        dnn_activation: str = "relu",
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

    def call(self, inputs, *args, **kwargs):
        filtered = self.filter_layer(inputs)
        output = self.dnn_layer(filtered)
        output = self.layernorm(self.dropout(output) + inputs)
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
    item_seq_feat_group : FeatureGroup
        _description_

    Returns
    -------
    Model
        _description_
    """

    embd_outputs = OrderedDict()
    pos_embd_outputs = OrderedDict()
    neg_embd_outputs = OrderedDict()
    negative_item_input = None
    for feat in seq_feat_group.features:
        if feat.seq_len % 2 == 1:
            raise AttributeError(
                f"`seq_len` of feature {feat.name} is odd, needs to be even"
            )
        sparse_embd = seq_feat_group.embd_layers[feat.unit.name]
        seq_input = seq_feat_group.input_layers[feat.name]
        embd_seq, mask = sparse_embd(seq_input)  # * (batch_size, seq_len, embd_dim)
        # * Get embeddings of positive and negative target items
        if negative_item_input is None:
            negative_item_input = Input(
                (1,), name="neg_" + feat.unit.id_input.name, dtype=tf.int32
            )
        embd_pos = sparse_embd.lookup(feat.unit.id_input)
        embd_neg = sparse_embd.lookup(negative_item_input)
        # TODO add support for SparseFeature
        pos_embd_outputs[feat.name] = embd_pos
        neg_embd_outputs[feat.name] = embd_neg

        # * Embedding Part (batch_size, seq_len, embd_dim)
        position_embd = PositionEmbedding()
        embd_seq = embd_seq + position_embd(embd_seq)
        embd_seq = embd_seq * tf.cast(mask, tf.float32)
        embd_seq = LayerNormalization(epsilon=layer_norm_eps)(embd_seq)
        embd_seq = Dropout(dropout)(embd_seq)

        # * Learnable Filter-enhanced Blocks
        for _ in range(block_num):
            embd_seq = FilterBlock(
                dnn_activation="relu",
                dropout=dropout,
                filter_dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )(embd_seq)

        # * (batch_size, embd_dim)
        embd_outputs[feat.name] = embd_seq[:, -1, :]

    # * Calculate loss
    output = concat([], list(embd_outputs.values()))
    pos_embd = concat([], list(pos_embd_outputs.values()))
    neg_embd = concat([], list(neg_embd_outputs.values()))

    pos_logits = tf.reduce_sum(output * pos_embd, axis=-1)
    neg_logits = tf.reduce_sum(output * neg_embd, axis=-1)
    loss = tf.reduce_mean(
        -tf.math.log(tf.sigmoid(pos_logits) + 1e-24)
        - tf.math.log(1 - tf.sigmoid(neg_logits) + 1e-24)
    )

    # * Construct model
    seq_feat_inputs = list(seq_feat_group.feat_pool.input_layers.values())
    inputs = [negative_item_input] + seq_feat_inputs
    model = Model(inputs=inputs, outputs=loss)
    model.add_loss(loss)

    model.__setattr__("actual_inputs", seq_feat_inputs)
    model.__setattr__("actual_outputs", pos_logits)

    return model


if __name__ == "__main__":
    import numpy as np
    from handyrec.features import (
        FeaturePool,
        FeatureGroup,
        EmbdFeatureGroup,
        SparseFeature,
        SparseSeqFeature,
    )

    item_dict = {
        "movie_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "genres": [
            [1, 2, 0],
            [1, 3, 0],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 0, 0],
            [22, 23, 0],
        ],
    }

    data_dict = {
        "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "hist_id": [
            [1, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [9, 0],
            [1, 3],
            [2, 0],
            [7, 9],
            [6, 8],
            [9, 0],
        ],
        "movie_id": [2, 3, 6, 7, 8, 4, 8, 9, 8, 10],
        "neg_movie_id": [5, 6, 8, 9, 10, 6, 3, 11, 9, 8],
    }
    data_dict = {x: np.array(y) for x, y in data_dict.items()}

    fp = FeaturePool()

    item_fg = EmbdFeatureGroup(
        name="movie",
        id_name="movie_id",
        features=[
            SparseFeature("movie_id", 10, 16),
            SparseSeqFeature(SparseFeature("genre_id", 23, 16), "genres", 3),
        ],
        feature_pool=fp,
        value_dict=item_dict,
        embd_dim=16,
    )

    seq_feat_group = FeatureGroup(
        "item_seq", [SparseSeqFeature(item_fg, "hist_id", 2)], fp
    )

    model = FMLPRec(
        seq_feat_group,
        dropout=0.1,
        block_num=4,
    )

    model.compile(optimizer="adam", loss=None)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="./rank_checkpoint/",
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    model.fit(data_dict, y=None, batch_size=1, epochs=3, callbacks=[checkpoint])
