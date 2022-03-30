from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Model

from handyrec.features import FeatureGroup, EmbdFeatureGroup
from handyrec.layers import DNN, SampledSoftmaxLayer
from handyrec.layers.utils import concat


def YouTubeMatchDNN(
    user_feature_group: FeatureGroup,
    item_feature_group: EmbdFeatureGroup,
    num_sampled: int = 1,
    dnn_hidden_units: Tuple[int] = (64, 32),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    seed: int = 2022,
) -> Model:
    """Implementation of YoutubeDNN match model

    Parameters
    ----------
    user_feature_group : FeatureGroup
        User feature group
    item_feature_group : EmbdFeatureGroup
        Item feature group
    num_sampled : int, optional
        Number of negative smaples in SampledSoftmax, by default `1`.
    dnn_hidden_units : Tuple[int], optional
        DNN structure, by default `(64, 32)`.
    dnn_activation : str, optional
        DNN activation function, by default `"relu"`.
    dnn_dropout : float, optional
        DNN dropout ratio, by default `0`.
    dnn_bn : bool, optional
        Whether use batch normalization or not, by default `False`.
    l2_dnn : float, optional
        DNN l2 regularization param, by default `0`.
    seed : int, optional
        Random seed of dropout, by default `2022`.

    Note
    ----
    The true structure of DNN is (*dnn_hidden_units, item embedding dim).

    Returns
    -------
    Model
        YouTubeDNN Match Model.

    Raises
    ------
    ValueError
        If `item_feature_group` is not an instance of `EmbdFeatureGroup`.
    """
    if not isinstance(item_feature_group, EmbdFeatureGroup):
        raise ValueError(
            "Item feature group should be an instance of `EmbdFeatureGroup`!"
        )

    user_dense, user_sparse = user_feature_group.embedding_lookup(pool_method="mean")

    user_dnn_input = concat(user_dense, user_sparse)
    item_id = item_feature_group.id_input
    full_item_embd = item_feature_group.get_embd(item_id)

    dnn_hidden_units = list(dnn_hidden_units) + [full_item_embd.shape[-1]]
    user_dnn_output = DNN(
        hidden_units=dnn_hidden_units,
        activation=dnn_activation,
        output_activation="linear",
        l2_reg=l2_dnn,
        use_bn=dnn_bn,
        dropout_rate=dnn_dropout,
        seed=seed,
    )(user_dnn_input)

    # * Sampled softmax output
    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [full_item_embd, user_dnn_output, item_id]
    )

    # * Construct model
    user_input = list(user_feature_group.input_layers.values())

    model = Model(inputs=user_input + [item_id], outputs=output)
    model.__setattr__("user_input", user_input)
    model.__setattr__("user_embedding", user_dnn_output)
    model.__setattr__("item_input", item_id)
    model.__setattr__("item_embedding", item_feature_group.lookup(item_id))

    return model
