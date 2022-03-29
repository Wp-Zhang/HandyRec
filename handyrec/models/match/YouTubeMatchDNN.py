import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda
from typing import OrderedDict, Tuple, List, Any

from handyrec.features import SparseFeature
from handyrec.features.utils import split_features
from handyrec.layers import (
    SequencePoolingLayer,
    DNN,
    ValueTable,
    SampledSoftmaxLayer,
)
from handyrec.layers.utils import (
    construct_input_layers,
    construct_embedding_layers,
    concat,
)


def YouTubeMatchDNN(
    user_features: List[Any],
    item_id: SparseFeature,
    num_sampled: int = 1,
    dnn_hidden_units: Tuple[int] = (64, 32),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    l2_emb: float = 1e-6,
    seed: int = 2022,
) -> Model:
    """Implementation of YoutubeDNN match model

    Args:
        user_features (List[Any]): user feature list
        item_id (SparseFeature): item id
        num_sampled (int, optional): number of negative smaples in SampledSoftmax. Defaults to 1.
        dnn_hidden_units (Tuple[int], optional): DNN structure. Defaults to (64, 32).
        dnn_activation (str, optional): DNN activation function. Defaults to "relu".
        dnn_dropout (float, optional): DNN dropout ratio. Defaults to 0.
        dnn_bn (bool, optional): whether use batch normalization or not. Defaults to False.
        l2_dnn (float, optional): DNN l2 regularization param. Defaults to 0.
        l2_emb (float, optional): embedding l2 regularization param. Defaults to 1e-6.
        seed (int, optional): random seed of dropout. Defaults to 2022.

    Raises:
        ValueError: length of `user_features` should be larger than 0

    Returns:
        Model: YouTubeDNN Match Model
    """
    if len(user_features) < 1:
        raise ValueError("Should have at least one user feature")
    if dnn_hidden_units[-1] != item_id.embdding_dim:
        raise ValueError("user DNN output dim should be equal with item embd dim")

    # * Group user features by their types
    u_dense, u_sparse, u_sparse_seq = split_features(user_features)

    # * Get input and embedding layers
    input_layers = construct_input_layers(user_features + [item_id])
    embd_layers = construct_embedding_layers(user_features + [item_id], l2_emb)

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

    # * Get full item embedding
    item_id_input = input_layers[item_id.name]
    item_index = ValueTable(list(range(item_id.vocab_size)))(item_id_input)
    full_item_embd = embd_layers[item_id.name](item_index)

    # * Concat input layers -> DNN
    user_dnn_input = concat(
        [input_layers[k] for k in u_dense.keys()], list(user_embd_outputs.values())
    )
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
        [full_item_embd, user_dnn_output, item_id_input]
    )

    # * Setup user/item input and embedding
    user_inputs = (
        list(u_dense.keys()) + list(u_sparse.keys()) + list(u_sparse_seq.keys())
    )
    user_inputs = [input_layers[f] for f in user_inputs]
    item_embedding = tf.nn.embedding_lookup(full_item_embd, item_id_input)
    item_embedding = tf.squeeze(item_embedding, axis=1)

    # * Construct model
    model = Model(inputs=list(input_layers.values()), outputs=output)
    model.__setattr__("user_input", user_inputs)
    model.__setattr__("user_embedding", user_dnn_output)
    model.__setattr__("item_input", item_id_input)
    model.__setattr__("item_embedding", item_embedding)

    return model
