from typing import OrderedDict, Tuple
from tensorflow.keras import Model
from handyrec.layers import SequencePoolingLayer, DNN
from handyrec.layers.utils import concat
from handyrec.features import FeatureGroup, EmbdFeatureGroup


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

    Args:
        dnn_hidden_units (Tuple[int], optional): DNN structure. Defaults to (64, 32).
        dnn_activation (str, optional): DNN activation function. Defaults to "relu".
        dnn_dropout (float, optional): DNN dropout ratio. Defaults to 0.
        dnn_bn (bool, optional): whether use batch normalization or not. Defaults to False.
        l2_dnn (float, optional): DNN l2 regularization param. Defaults to 0.
        seed (int, optional): random seed of dropout. Defaults to 2022.

    Raises:
        ValueError: length of `user_features` should be larger than 0

    Returns:
        Model: YouTubeDNN Match Model
    """
    user_dense = user_feature_group.dense_output
    user_sparse = user_feature_group.sparse_output

    item_dense = item_feature_group.dense_output
    item_sparse = item_feature_group.sparse_output

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
