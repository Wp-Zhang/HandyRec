from typing import Tuple
from tensorflow.keras import Model
from handyrec.layers import DNN
from handyrec.layers.utils import concat
from handyrec.features import FeatureGroup


def YouTubeRankDNN(
    user_feature_group: FeatureGroup,
    item_feature_group: FeatureGroup,
    dnn_hidden_units: Tuple[int] = (64, 32),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    seed: int = 2022,
) -> Model:
    """Implementation of YoutubeDNN rank model

    Parameters
    ----------
    user_feature_group : FeatureGroup
        User feature group.
    item_feature_group : FeatureGroup
        Item feature group.
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
    seed : int, optional
        Random seed of dropout, by default ``2022``.

    Returns
    -------
    Model
        A YouTubeDNN rank mdoel.

    References
    ----------
    .. [1] Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube
        recommendations." Proceedings of the 10th ACM conference on recommender systems. 2016.
    """
    user_dense, user_sparse = user_feature_group.embedding_lookup(pool_method="mean")
    item_dense, item_sparse = item_feature_group.embedding_lookup(pool_method="mean")

    # * concat input layers -> DNN
    dnn_input = concat(user_dense + item_dense, user_sparse + item_sparse)
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
    user_input = list(user_feature_group.input_layers.values())
    item_input = list(item_feature_group.input_layers.values())
    model = Model(inputs=user_input + item_input, outputs=dnn_output)

    return model
