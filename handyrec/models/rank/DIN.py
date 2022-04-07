from typing import Tuple, OrderedDict
import warnings
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from handyrec.features import FeatureGroup, EmbdFeatureGroup
from handyrec.layers import DNN, FM, LocalActivationUnit
from handyrec.layers.utils import concat


def DIN(
    item_seq_feat_group: EmbdFeatureGroup,
    other_feature_group: FeatureGroup,
    dnn_hidden_units: Tuple[int] = (64, 32, 1),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    lau_dnn_hidden_units: Tuple[int] = (64, 32, 1),
    lau_dnn_activation: str = "relu",
    lau_dnn_dropout: float = 0,
    lau_dnn_bn: bool = False,
    lau_l2_dnn: float = 0,
    seed: int = 2022,
) -> Model:
    """Implementation of YoutubeDNN rank model

    Parameters
    ----------
    item_seq_feat_group : EmbdFeatureGroup
        Item sequence feature group.
    other_feature_group : FeatureGroup
        Feature group for other features.
    dnn_hidden_units : Tuple[int], optional
        DNN structure, by default ``(64, 32)``.
    dnn_activation : str, optional
        DNN activation function, by default ``"relu"``.
    dnn_dropout : float, optional
        DNN dropout ratio, by default ``0``.
    dnn_bn : bool, optional
        Whether use batch normalization or not, by default ``False``.
    l2_dnn : float, optional
        DNN l2 regularization param, by default ``0``.
    lau_dnn_hidden_units : Tuple[int], optional
        DNN structure in local activation unit, by default ``(64, 32)``.
    lau_dnn_activation : str, optional
        DNN activation function in local activation unit, by default ``"relu"``.
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
    .. [1] Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube
        recommendations." Proceedings of the 10th ACM conference on recommender systems. 2016.
    """
    other_dense, other_sparse = other_feature_group.embedding_lookup(pool_method="mean")

    embd_outputs = OrderedDict()
    for feat in item_seq_feat_group.features:
        sparse_embd = item_seq_feat_group.embd_layers[feat.unit.name]
        seq_input = item_seq_feat_group.input_layers[feat.name]
        local_activate = LocalActivationUnit(
            dnn_hidden_units, dnn_activation, l2_dnn, dnn_dropout, dnn_bn, seed
        )
        embd_seq, mask = sparse_embd(seq_input)  # * (batch_size, seq_len, embd_dim)
        # * att_score: (batch_size, 1, seq_len)
        query = tf.expand_dims(feat.unit.lookup(feat.unit.id_input), axis=1)
        att_score = local_activate([query, embd_seq])
        mask = tf.cast(tf.expand_dims(mask[:, :, 0], axis=1), dtype=tf.float32)
        att_score *= mask
        # * (batch_size, 1, embd_dim)
        print(att_score.shape)
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
    feature_pool = item_seq_feat_group.feat_pool
    # other_input = list(other_feature_group.input_layers.values())
    # item_seq_input = list(item_seq_feat_group.input_layers.values())
    model = Model(inputs=list(feature_pool.input_layers.values()), outputs=dnn_output)

    return model
