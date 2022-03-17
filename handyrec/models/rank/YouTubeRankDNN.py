from tensorflow.keras import Model
from typing import OrderedDict, Tuple, List, Any

from ...features.utils import split_features
from ...layers import SequencePoolingLayer, DNN
from ...layers.utils import (
    construct_input_layers,
    construct_embedding_layers,
    concat_inputs,
)


def YouTubeRankDNN(
    user_features: List[Any],
    item_features: List[Any],
    dnn_hidden_units: Tuple[int] = (64, 32),
    dnn_activation: str = "relu",
    l2_dnn: float = 0,
    l2_emb: float = 1e-6,
    dnn_dropout: float = 0,
    seed: int = 2022,
) -> Model:
    """Implementation of YoutubeDNN rank model

    Args:
        user_features (List[Any]): user feature list
        item_features (List[Any]): item feature list
        dnn_hidden_units (Tuple[int], optional): DNN structure. Defaults to (64, 32).
        dnn_activation (str, optional): DNN activation function. Defaults to "relu".
        l2_dnn (float, optional): DNN l2 regularization param. Defaults to 0.
        l2_emb (float, optional): embedding l2 regularization param. Defaults to 1e-6.
        dnn_dropout (float, optional): DNN dropout ratio. Defaults to 0.
        seed (int, optional): random seed of dropout. Defaults to 2022.

    Raises:
        ValueError: length of `user_features` should be larger than 0

    Returns:
        Model: YouTubeDNN Match Model
    """
    if len(user_features) < 1:
        raise ValueError("Should have at least one user feature")

    u_dense, u_sparse, u_sparse_seq = split_features(user_features)
    i_dense, i_sparse, i_sparse_seq = split_features(item_features)

    # * Get input and embedding layers
    input_layers = construct_input_layers(user_features + item_features)
    embd_layers = construct_embedding_layers(user_features + item_features, l2_emb)

    # * Embedding output: input layer -> embedding layer (-> pooling layer)
    user_embd_outputs = OrderedDict()
    for feat in u_sparse.keys():
        user_embd_outputs[feat] = embd_layers[feat](input_layers[feat])
    for feat in u_sparse_seq.values():
        sparse_emb = embd_layers[feat.sparse_feat.name]
        seq_input = input_layers[feat.name]
        user_embd_outputs[feat.name] = SequencePoolingLayer("mean")(
            sparse_emb(seq_input)
        )
    item_embd_outputs = OrderedDict()
    for feat in i_sparse.keys():
        item_embd_outputs[feat] = embd_layers[feat](input_layers[feat])
    for feat in i_sparse_seq.values():
        sparse_emb = embd_layers[feat.sparse_feat.name]
        seq_input = input_layers[feat.name]
        item_embd_outputs[feat.name] = SequencePoolingLayer("mean")(
            sparse_emb(seq_input)
        )

    # * concat input layers -> DNN
    dnn_input = concat_inputs(
        [input_layers[k] for k in list(u_dense.keys()) + list(i_dense.keys())],
        list(user_embd_outputs.values()) + list(item_embd_outputs.values()),
    )
    dnn_output = DNN(
        hidden_units=tuple(list(dnn_hidden_units) + [1]),
        activation=dnn_activation,
        output_activation="sigmoid",
        l2_reg=l2_dnn,
        dropout_rate=dnn_dropout,
        seed=seed,
    )(dnn_input)

    # * Construct model
    model = Model(inputs=list(input_layers.values()), outputs=dnn_output)

    return model
