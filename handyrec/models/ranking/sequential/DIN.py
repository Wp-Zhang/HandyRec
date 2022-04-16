from typing import Tuple, OrderedDict
import tensorflow as tf
from tensorflow.keras import Model
from handyrec.features import FeatureGroup, EmbdFeatureGroup
from handyrec.layers import DNN, LocalActivationUnit, SqueezeMask
from handyrec.layers.utils import concat


def DIN(
    item_seq_feat_group: EmbdFeatureGroup,
    other_feature_group: FeatureGroup,
    dnn_hidden_units: Tuple[int] = (64, 32, 1),
    dnn_activation: str = "dice",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    lau_dnn_hidden_units: Tuple[int] = (32, 1),
    lau_dnn_activation: str = "dice",
    lau_dnn_dropout: float = 0,
    lau_dnn_bn: bool = False,
    lau_l2_dnn: float = 0,
    seed: int = 2022,
) -> Model:
    """Implementation of Deep Interest Network (DIN) model

    Parameters
    ----------
    item_seq_feat_group : EmbdFeatureGroup
        Item sequence feature group.
    other_feature_group : FeatureGroup
        Feature group for other features.
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
    lau_dnn_hidden_units : Tuple[int], optional
        DNN structure in local activation unit, by default ``(32, 1)``.
    lau_dnn_activation : str, optional
        DNN activation function in local activation unit, by default ``"dice"``.
    lau_dnn_dropout : float, optional
        DNN dropout ratio in local activation unit, by default ``0``.
    lau_dnn_bn : bool, optional
        Whether use batch normalization or not in local activation unit, by default ``False``.
    l2_dnn : float, optional
        DNN l2 regularization param in local activation unit, by default ``0``.
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
    feature_pool = item_seq_feat_group.feat_pool
    other_dense, other_sparse = other_feature_group.embedding_lookup(pool_method="mean")

    embd_outputs = OrderedDict()
    id_input = None
    for feat in item_seq_feat_group.features:
        if id_input is None:
            id_input = feature_pool.init_input(
                feat.unit.name,
                {"name": feat.unit.name, "shape": (1,), "dtype": tf.int32},
            )
        sparse_embd = item_seq_feat_group.embd_layers[feat.unit.name]
        seq_input = item_seq_feat_group.input_layers[feat.name]
        lau = LocalActivationUnit(
            lau_dnn_hidden_units,
            lau_dnn_activation,
            lau_l2_dnn,
            lau_dnn_dropout,
            lau_dnn_bn,
            seed,
        )
        embd_seq = sparse_embd(seq_input)  # * (batch_size, seq_len, embd_dim)
        embd_seq = SqueezeMask()(embd_seq)
        # * att_score: (batch_size, 1, seq_len)
        query = sparse_embd(id_input)
        att_score = lau([query, embd_seq])
        # * (batch_size, 1, embd_dim)
        embd_outputs[feat.name] = tf.matmul(att_score, embd_seq)
    local_activate_pool = list(embd_outputs.values())

    # * concat input layers -> DNN
    dnn_input = concat(other_dense, other_sparse + local_activate_pool)
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
    inputs = list(feature_pool.input_layers.values())
    model = Model(inputs=inputs, outputs=dnn_output)

    return model
