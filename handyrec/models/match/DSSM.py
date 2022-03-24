from typing import OrderedDict, Tuple, List, Any
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.losses import cosine_similarity

from handyrec.features.utils import split_features
from handyrec.layers import SequencePoolingLayer, DNN, Similarity
from handyrec.layers.utils import (
    construct_input_layers,
    construct_embedding_layers,
    concat_inputs,
)


def DSSM(
    user_features: List[Any],
    item_features: List[Any],
    user_dnn_hidden_units: Tuple[int] = (64, 32),
    item_dnn_hidden_units: Tuple[int] = (64, 32),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    l2_emb: float = 1e-6,
    gamma: float = 0.2,
    seed: int = 2022,
):
    """Implemetation of the classic two tower model originated from DSSM.

    Args:
        user_features (List[Any]): user feature list
        item_features (List[Any]): item feature list
        user_dnn_hidden_units (Tuple[int], optional): user DNN structure. Defaults to (64, 32).
        item_dnn_hidden_units (Tuple[int], optional): item DNN structure. Defaults to (64, 32).
        dnn_activation (str, optional): DNN activation function. Defaults to "relu".
        dnn_dropout (float, optional): DNN dropout ratio. Defaults to 0.
        dnn_bn (bool, optional): whether to use batch normalization. Defaults to False.
        l2_dnn (float, optional): DNN l2 regularization param. Defaults to 0.
        l2_emb (float, optional): embedding l2 regularization param. Defaults to 1e-6.
        gamma (float, optional): smoothing factor for softmax mention in DSSM paper chapter 3.3. Defaults to 0.2.
        seed (int, optional): random seed of dropout. Defaults to 2022.
    """
    if len(user_features) < 1:
        raise ValueError("Should have at least one user feature")
    if len(item_features) < 1:
        raise ValueError("Should have at least one item feature")

    # * Group features by their types
    u_dense_f, u_sparse_f, u_sparse_seq_f = split_features(user_features)
    i_dense_f, i_sparse_f, i_sparse_seq_f = split_features(item_features)
    _, sparse_f, sparse_seq_f = split_features(user_features + item_features)

    # * Get input and embedding layers
    input_layers = construct_input_layers(user_features + item_features)
    embd_layers = construct_embedding_layers(user_features + item_features, l2_emb)

    # * Embedding output: input layer -> embedding layer (-> pooling layer)
    embd_outputs = OrderedDict()
    for feat in sparse_f.keys():
        embd_outputs[feat] = embd_layers[feat](input_layers[feat])
    for feat in sparse_seq_f.values():
        sparse_emb = embd_layers[feat.sparse_feat.name]
        seq_input = input_layers[feat.name]
        embd_outputs[feat.name] = SequencePoolingLayer("mean")(sparse_emb(seq_input))

    # * Concat input layers -> DNN
    u_dnn_input = concat_inputs(
        [input_layers[k] for k in u_dense_f.keys()],
        [embd_outputs[k] for k in u_sparse_f.keys()]
        + [embd_outputs[k] for k in u_sparse_seq_f.keys()],
    )
    i_dnn_input = concat_inputs(
        [input_layers[k] for k in i_dense_f.keys()],
        [embd_outputs[k] for k in i_sparse_f.keys()]
        + [embd_outputs[k] for k in i_sparse_seq_f.keys()],
    )

    u_embedding = DNN(
        hidden_units=user_dnn_hidden_units,
        activation=dnn_activation,
        l2_reg=l2_dnn,
        dropout_rate=dnn_dropout,
        use_bn=dnn_bn,
        output_activation="linear",
        seed=seed,
    )(u_dnn_input)

    i_embedding = DNN(
        hidden_units=item_dnn_hidden_units,
        activation=dnn_activation,
        l2_reg=l2_dnn,
        dropout_rate=dnn_dropout,
        use_bn=dnn_bn,
        output_activation="linear",
        seed=seed,
    )(i_dnn_input)

    # * Output
    # output = cosine_similarity(u_embedding, i_embedding, axis=-1) * gamma
    output = Similarity("cos")([u_embedding, i_embedding]) * gamma
    # output = tf.reshape(output, (-1, 1))
    output = Activation("softmax")(output)

    # * Construct model
    user_inputs = [input_layers[f.name] for f in user_features]
    item_inputs = [input_layers[f.name] for f in item_features]
    model = Model(inputs=list(input_layers.values()), outputs=output)
    model.__setattr__("user_input", user_inputs)
    model.__setattr__("user_embedding", u_embedding)
    model.__setattr__("item_input", item_inputs)
    model.__setattr__("item_embedding", i_embedding)

    return model
