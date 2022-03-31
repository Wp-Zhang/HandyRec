from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Model

from handyrec.features import FeatureGroup, EmbdFeatureGroup
from handyrec.layers import DNN, SampledSoftmaxLayer
from handyrec.layers.utils import concat


def DSSM(
    user_feature_group: FeatureGroup,
    item_feature_group: EmbdFeatureGroup,
    user_dnn_hidden_units: Tuple[int] = (64, 32),
    item_dnn_hidden_units: Tuple[int] = (64, 32),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    num_sampled: int = 1,
    seed: int = 2022,
    cos_sim: bool = False,
    gamma: float = 10,
) -> Model:
    """Implemetation of the classic two tower model originated from DSSM.

    Parameters
    ----------
    user_feature_group : FeatureGroup
        User feature group.
    item_feature_group : EmbdFeatureGroup
        Item feature group.
    user_dnn_hidden_units : Tuple[int], optional
        User DNN structure, by default `(64, 32)`.
    item_dnn_hidden_units : Tuple[int], optional
        Item DNN structure, by default `(64, 32)`.
    dnn_activation : str, optional
        DNN activation function, by default `"relu"`.
    dnn_dropout : float, optional
        DNN dropout ratio, by default `0`.
    dnn_bn : bool, optional
        Whether use batch normalization or not, by default `False`.
    l2_dnn : float, optional
        DNN l2 regularization param, by default `0`.
    num_sampled : int, optional
        Number of negative smaples in SampledSoftmax, by default `1`.
    seed : int, optional
        Random seed of dropout, by default `2022`.
    cos_sim : bool, optional
        Whether use cosine similarity or not, by default `False`.
    gamma : float, optional
        Smoothing factor for cosine similarity softmax, by default `10`.

    Returns
    -------
    Model
        A DSSM model.

    Raises
    ------
    ValueError
        If `item_feature_group` is not an instance of `EmbdFeatureGroup`.
    """
    if not isinstance(item_feature_group, EmbdFeatureGroup):
        raise ValueError(
            "Item feature group should be an instance of `EmbdFeatureGroup`!"
        )

    user_dense, user_sparse = user_feature_group.embedding_lookup(pool_method="sum")
    user_dnn_input = concat(user_dense, user_sparse)

    item_id = item_feature_group.id_input
    full_item_embd = item_feature_group.get_embd(item_id)

    user_embedding = DNN(
        hidden_units=user_dnn_hidden_units,
        activation=dnn_activation,
        l2_reg=l2_dnn,
        dropout_rate=dnn_dropout,
        use_bn=dnn_bn,
        output_activation="linear",
        seed=seed,
    )(user_dnn_input)

    item_embedding = DNN(
        hidden_units=item_dnn_hidden_units,
        activation=dnn_activation,
        l2_reg=l2_dnn,
        dropout_rate=dnn_dropout,
        use_bn=dnn_bn,
        output_activation="linear",
        seed=seed,
    )(full_item_embd)

    # * Sampled cosine similarity softmax output
    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [
            tf.nn.l2_normalize(item_embedding) * gamma if cos_sim else item_embedding,
            tf.nn.l2_normalize(user_embedding) if cos_sim else user_embedding,
            item_id,
        ]
    )

    # * Construct model
    user_inputs = list(user_feature_group.input_layers.values())
    item_embdding = tf.nn.embedding_lookup(item_embedding, item_id)
    item_embdding = tf.squeeze(item_embdding, axis=1)

    model = Model(inputs=user_inputs + [item_id], outputs=output)
    model.__setattr__("user_input", user_inputs)
    model.__setattr__("user_embedding", user_embedding)
    model.__setattr__("item_input", item_id)
    model.__setattr__("item_embedding", item_embdding)

    return model
