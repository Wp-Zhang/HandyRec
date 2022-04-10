from typing import Tuple, OrderedDict
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GRU, Concatenate, RNN
from handyrec.features import FeatureGroup, EmbdFeatureGroup
from handyrec.layers import DNN, LocalActivationUnit, AUGRUCell
from handyrec.layers.utils import concat


def _auxiliary_loss(embd_seq, neg_embd_seq, hidden_seq, mask):
    mask = tf.cast(mask, tf.float32)
    embd_seq = embd_seq[:, 1:, :]
    neg_embd_seq = neg_embd_seq[:, 1:, :]
    hidden_seq = hidden_seq[:, :-1, :]
    concat_seq1 = Concatenate(axis=-1)([hidden_seq, neg_embd_seq])
    concat_seq2 = Concatenate(axis=-1)([hidden_seq, embd_seq])

    dnn = DNN(
        hidden_units=(100, 50, 1),
        activation="sigmoid",
        use_bn=True,
        output_activation="sigmoid",
    )
    click_p = tf.squeeze(dnn(concat_seq1))  # * (batch_size, seq_len-1)
    nonclick_p = tf.squeeze(dnn(concat_seq2))  # * (batch_size, seq_len-1)
    click_loss = tf.math.log(click_p) * mask
    nonclick_loss = tf.math.log(nonclick_p) * mask

    loss = tf.reduce_sum(click_loss + nonclick_loss, axis=1)
    loss = -tf.reduce_mean(loss)

    return loss


def DIEN(
    item_seq_feat_group: FeatureGroup,
    neg_item_seq_feat_group: FeatureGroup,
    other_feature_group: FeatureGroup,
    # * =================================================
    gru_units: int = 8,
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
    augru_dropout: float = 0,
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
    gru_units: int
        GRU units in first layer, by default ``64``.
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
    lau_dnn_dropout : float, optional
        DNN dropout ratio in local activation unit, by default ``0``.
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
    augru_dropout: float
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
        A YouTubeDNN rank mdoel.

    References
    ----------
    .. [1] Zhou, Guorui, et al. "Deep interest network for click-through rate prediction."
        Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery
        & data mining. 2018.
    """
    item_seq_feat_names = sorted([x.name for x in item_seq_feat_group.features])
    neg_item_seq_feat_names = sorted([x.name for x in neg_item_seq_feat_group.features])
    if ["neg_" + x for x in item_seq_feat_names] != neg_item_seq_feat_names:
        raise ValueError(
            """`item_seq_feat_group` and `neg_item_seq_feat_group` should have """
            """the same features. Features in `neg_item_seq_feat_group` should """
            """be prefixed with `neg_`."""
        )
    # TODO check seq length equality

    other_dense, other_sparse = other_feature_group.embedding_lookup(pool_method="mean")

    embd_outputs = OrderedDict()
    for feat in item_seq_feat_group.features:
        sparse_embd = item_seq_feat_group.embd_layers[feat.unit.name]
        seq_input = item_seq_feat_group.input_layers[feat.name]
        neg_seq_input = neg_item_seq_feat_group.input_layers["neg_" + feat.name]
        # * the input sequence is in descending order, so we need to reverse it
        seq_input = seq_input[:, ::-1]
        neg_seq_input = neg_seq_input[:, ::-1]

        # * ========================== Embedding Lookup ==========================

        embd_seq, mask = sparse_embd(seq_input)  # * (batch_size, seq_len, embd_dim)
        neg_embd_seq, _ = sparse_embd(neg_seq_input)
        mask = mask[:, :, 0]  # * (batch_size, seq_len)

        # * ========================== FIRST LAYER: GRU ==========================
        gru = GRU(
            gru_units,
            activation=gru_activation,
            recurrent_activation=gru_recurrent_activation,
            dropout=gru_dropout,
            return_sequences=True,
            return_state=False,
            name="gru",
        )
        hidden_seq1 = gru(embd_seq, mask=mask)  # * (batch_size, seq_len, gru_units)
        auxiliary_loss = _auxiliary_loss(embd_seq, neg_embd_seq, hidden_seq1, mask)

        # * ======================= LOCAL ACTIVATION UNIT =======================
        query = tf.expand_dims(feat.unit.lookup(feat.unit.id_input), axis=1)
        lau = LocalActivationUnit(
            lau_dnn_hidden_units,
            lau_dnn_activation,
            lau_l2_dnn,
            lau_dnn_dropout,
            lau_dnn_bn,
            seed,
        )
        att_score = lau([query, hidden_seq1])  # * att_score: (batch_size, 1, seq_len)
        att_score = att_score * tf.cast(mask, dtype=tf.float32)
        att_score = tf.transpose(att_score, [0, 2, 1])  # * (batch_size, seq_len, 1)
        # * ======================== SECOND LAYER: AUGRU ========================
        augru_cell = AUGRUCell(
            augru_units,
            activation=augru_activation,
            recurrent_activation=augru_recurrent_activation,
        )
        augru = RNN(cell=augru_cell, return_sequences=True, return_state=True)
        # * (batch_size, augru_units)
        _, final_state = augru((hidden_seq1, att_score), mask=mask)

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
    model = Model(inputs=list(feature_pool.input_layers.values()), outputs=dnn_output)
    model.add_loss(alpha * auxiliary_loss)

    return model
