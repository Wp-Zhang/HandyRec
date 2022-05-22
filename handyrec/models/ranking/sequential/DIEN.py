from typing import Tuple, OrderedDict
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU, RNN, Layer
from handyrec.features import FeatureGroup
from handyrec.layers import DNN, LocalActivationUnit, AUGRUCell, SqueezeMask
from handyrec.layers.utils import concat


class AuxiliaryLoss(Layer):
    def __init__(self, dnn_units: Tuple = (100, 50, 1), **kwargs):
        self.dnn = DNN(hidden_units=dnn_units, activation="sigmoid")
        super().__init__(**kwargs)

    def call(self, inputs, mask, **kwargs):
        embd_seq, neg_embd_seq, hidden_seq = inputs
        # * embd_seq: [batch_size, T, embd_size]
        # * neg_embd_seq: [batch_size, T, embd_size]
        # * hidden_seq: [batch_size, T, hidden_size]

        mask = tf.cast(mask[0], tf.float32)[:, 1:]  # * [batch_size, T-1]
        embd_seq = embd_seq[:, 1:, :]  # * [batch_size, T-1, embd_size]
        neg_embd_seq = neg_embd_seq[:, 1:, :]  # * [batch_size, T-1, embd_size]
        hidden_seq = hidden_seq[:, :-1, :]  # * [batch_size, T-1, hidden_size]
        concat_seq1 = tf.concat([hidden_seq, neg_embd_seq], axis=2)
        concat_seq2 = tf.concat([hidden_seq, embd_seq], axis=2)

        click_p = tf.squeeze(self.dnn(concat_seq1), axis=2)  # * (batch_size, T-1)
        nonclick_p = tf.squeeze(self.dnn(concat_seq2), axis=2)  # * (batch_size, T-1)
        click_p = tf.clip_by_value(tf.sigmoid(click_p), 1e-8, 1 - 1e-8)
        nonclick_p = tf.clip_by_value(tf.sigmoid(nonclick_p), 1e-8, 1 - 1e-8)
        click_loss = tf.math.log(click_p) * mask
        nonclick_loss = tf.math.log(1 - nonclick_p) * mask

        loss = tf.reduce_sum(click_loss + nonclick_loss, axis=1)
        loss = -tf.reduce_mean(loss, axis=0)

        return loss

    def get_config(self):
        base_config = super().get_config()
        config = self.dnn.get_config()
        return {**base_config, **config}


def DIEN(
    item_seq_feat_group: FeatureGroup,
    neg_item_seq_feat_group: FeatureGroup,
    other_feature_group: FeatureGroup,
    # * =================================================
    gru_activation: str = "tanh",
    gru_recurrent_activation: str = "sigmoid",
    gru_dropout: float = 0,
    # * =================================================
    lau_dnn_hidden_units: Tuple[int] = (32, 1),
    lau_dnn_activation: str = "dice",
    lau_dnn_dropout: float = 0,
    lau_dnn_bn: bool = False,
    lau_l2_dnn: float = 0,
    # * =================================================
    augru_units: int = 8,
    augru_activation: str = "tanh",
    augru_recurrent_activation: str = "sigmoid",
    alpha: float = 0.5,
    # * =================================================
    dnn_hidden_units: Tuple[int] = (64, 32, 1),
    dnn_activation: str = "dice",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    seed: int = 2022,
) -> Model:
    """Implementation of Deep Interest Evolution Network (DIEN) model

    Parameters
    ----------
    item_seq_feat_group : EmbdFeatureGroup
        Item sequence feature group.
    neg_item_seq_feat_group : EmbdFeatureGroup
        Negative item sequence feature group corresponding to `item_seq_feat_group`.
    other_feature_group : FeatureGroup
        Feature group for other features.
    gru_activation: str
        GRU activation function in first layer, by default ``"tanh"``.
    gru_recurrent_activation: str
        GRU recurrent activation function in first layer, by default ``"sigmoid"``.
    gru_dropout: float
        GRU dropout ratio in first layer, by default ``0``.
    lau_dnn_hidden_units : Tuple[int], optional
        DNN structure in local activation unit, by default ``(32, 1)``.
    lau_dnn_activation : str, optional
        DNN activation function in local activation unit, by default ``"dice"``.
    lau_dnn_bn : bool, optional
        Whether use batch normalization or not in local activation unit, by default ``False``.
    lau_l2_dnn : float, optional
        DNN l2 regularization param in local activation unit, by default ``0``.
    augru_units: int
        AUGRU units in second layer, by default ``32``.
    augru_activation: str
        AUGRU activation function in second layer, by default ``"tanh"``.
    augru_recurrent_activation: str
        AUGRU recurrent activation function in second layer, by default ``"sigmoid"``.
    augru_dropout: floa
        AUGRU dropout ratio in second layer, by default ``0``.
    alpha : float, optional
        Weight of auxiliary loss, by default ``0.5``.
    dnn_hidden_units : Tuple[int], optional
        DNN structure, by default ``(64, 32, 1)``.
    dnn_activation : str, optional
        DNN activation function, by default ``"dice"``.
    dnn_dropout : float, optional
        DNN dropout ratio, by default ``0``.
    dnn_bn : bool, optional
        Whether use batch normalization or not, by default ``False``.
    l2_dnn : float, optional
        DNN l2 regularization param, by default ``0``.
    seed : int, optional
        Random seed of dropout in local activation unit, by default ``2022``.

    Returns
    -------
    Model
        A DIEN mdoel.

    Note
    ----
    `item_seq_feat_group` and `neg_item_seq_feat_group` should have the same features, i.e. the same
        sequence length and featue name. Name of eatures in `neg_item_seq_feat_group` should be
        prefixed with ``neg_``.

    Raises
    ------
    ValueError
        If features in `neg_item_seq_feat_group` don't match with `item_seq_feat_group`.

    References
    ----------
    .. [1] Zhou, Guorui, et al. "Deep interest evolution network for click-through rate prediction."
        Proceedings of the AAAI conference on artificial intelligence. Vol. 33. No. 01. 2019.
    """
    seq_feat_dict = {x.name: x for x in item_seq_feat_group.features}
    neg_seq_feat_dict = {x.name: x for x in neg_item_seq_feat_group.features}
    for feat in seq_feat_dict.keys():
        neg_feat = neg_seq_feat_dict.get("neg_" + feat, None)
        if neg_feat is None or neg_feat.seq_len != seq_feat_dict[feat].seq_len:
            raise ValueError(
                "`item_seq_feat_group` and `neg_item_seq_feat_group` should have the same"
                " features, i.e. the same sequence length and featue name. Name of features"
                " in `neg_item_seq_feat_group` should be prefixed with `neg_`."
            )
    # TODO check units of SparseSeqFeats are the same

    feature_pool = item_seq_feat_group.feat_pool
    other_dense, other_sparse = other_feature_group.embedding_lookup(pool_method="mean")

    embd_outputs = OrderedDict()
    id_input = None
    for feat in item_seq_feat_group.features:
        # * Due to the feature of `FeaturePool`, `item_seq_feat_group` and
        # *     `neg_item_seq_feat_group` actually share the same embedding
        # *     layers, so we can use embedding layer dict in `item_seq_feat_group`
        # *     to lookup embedding of unit features in `neg_item_seq_feat_group`.
        sparse_embd = item_seq_feat_group.embd_layers[feat.unit.name]
        if id_input is None:
            id_input = feature_pool.init_input(
                feat.unit.name,
                {"name": feat.unit.name, "shape": (1,), "dtype": tf.int32},
            )

        seq_input = item_seq_feat_group.input_layers[feat.name]
        neg_seq_input = neg_item_seq_feat_group.input_layers["neg_" + feat.name]
        # # * the input sequence is in descending order, so we need to reverse it
        # seq_input = seq_input[:, ::-1]
        # neg_seq_input = neg_seq_input[:, ::-1]

        # * ========================== Embedding Lookup ==========================
        embd_seq = sparse_embd(seq_input)  # * (batch_size, seq_len, embd_dim)
        neg_embd_seq = sparse_embd(neg_seq_input)
        # * layers below only use the mask of `embd_seq`, thus we only apply
        # * squeeze operation on it.
        embd_seq = SqueezeMask()(embd_seq)

        # * ========================== FIRST LAYER: GRU ==========================
        gru = GRU(
            embd_seq.shape[-1],
            activation=gru_activation,
            recurrent_activation=gru_recurrent_activation,
            dropout=gru_dropout,
            return_sequences=True,
            return_state=False,
            name="gru",
        )
        hidden_seq1 = gru(embd_seq)  # * (batch_size, seq_len, gru_units)
        auxiliary_loss = AuxiliaryLoss()([embd_seq, neg_embd_seq, hidden_seq1])

        # * ======================= LOCAL ACTIVATION UNIT =======================
        query = sparse_embd(id_input)

        lau = LocalActivationUnit(
            lau_dnn_hidden_units,
            lau_dnn_activation,
            lau_l2_dnn,
            lau_dnn_dropout,
            lau_dnn_bn,
            seed,
        )
        # * att_score: (batch_size, 1, seq_len)
        att_score = lau([query, hidden_seq1])
        att_score = tf.transpose(att_score, [0, 2, 1])  # * (batch_size, seq_len, 1)

        # * ======================== SECOND LAYER: AUGRU ========================
        augru_cell = AUGRUCell(
            augru_units,
            activation=augru_activation,
            recurrent_activation=augru_recurrent_activation,
        )
        augru = RNN(cell=augru_cell, return_sequences=True, return_state=True)
        # * (batch_size, augru_units)
        _, final_state = augru((hidden_seq1, att_score))

        # * =====================================================================
        # * shape: (batch_size, 1, augru_units)
        embd_outputs[feat.name] = tf.expand_dims(final_state, axis=1)

    # * concat input layers -> DNN
    lau_pooled = list(embd_outputs.values())
    dnn_input = concat(other_dense, other_sparse + lau_pooled)
    dnn_output = DNN(
        hidden_units=tuple(list(dnn_hidden_units) + [1]),
        activation=dnn_activation,
        output_activation="sigmoid",
        l2_reg=l2_dnn,
        dropout_rate=dnn_dropout,
        use_bn=dnn_bn,
        seed=seed,
    )(dnn_input)

    # * Construct model
    feature_pool = item_seq_feat_group.feat_pool
    inputs = list(feature_pool.input_layers.values())
    model = Model(inputs=inputs, outputs=dnn_output)
    model.add_loss(alpha * auxiliary_loss)

    item_seq_inputs = set(item_seq_feat_group.input_layers.keys())
    other_inputs = set(other_feature_group.input_layers.keys())
    real_input_names = list(item_seq_inputs | other_inputs)
    real_inputs = [feature_pool.input_layers[x] for x in real_input_names]
    model.__setattr__("real_inputs", real_inputs)

    return model
