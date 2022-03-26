from typing import OrderedDict, Tuple, List, Any, Dict
import warnings
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.losses import cosine_similarity

from handyrec.features.utils import split_features
from handyrec.layers import (
    SequencePoolingLayer,
    DNN,
    Similarity,
    ValueTable,
    SampledSoftmaxLayer,
)
from handyrec.layers.utils import (
    construct_input_layers,
    construct_embedding_layers,
    concat_inputs,
)


def DSSM(
    user_features: List[Any],
    item_features: List[Any],
    item_id_name: str,
    full_item_dict: Dict,
    user_dnn_hidden_units: Tuple[int] = (64, 32),
    item_dnn_hidden_units: Tuple[int] = (64, 32),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    l2_emb: float = 1e-6,
    num_sampled: int = 1,
    seed: int = 2022,
    cos_sim: bool = False,
    gamma: float = 1,
):
    """Implemetation of the classic two tower model originated from DSSM.

    Args:
        user_features (List[Any]): user feature list
        item_features (List[Any]): item feature list
        item_id_name (str): name of item id,
        full_item_dict (Dict): full item feature map dict, {feature name:value array}
        user_dnn_hidden_units (Tuple[int], optional): user DNN structure. Defaults to (64, 32).
        item_dnn_hidden_units (Tuple[int], optional): item DNN structure. Defaults to (64, 32).
        dnn_activation (str, optional): DNN activation function. Defaults to "relu".
        dnn_dropout (float, optional): DNN dropout ratio. Defaults to 0.
        dnn_bn (bool, optional): whether use batch normalization or not. Defaults to False.
        l2_dnn (float, optional): DNN l2 regularization param. Defaults to 0.
        l2_emb (float, optional): embedding l2 regularization param. Defaults to 1e-6.
        num_sampled (int, optional): number of negative smaples in SampledSoftmax. Defaults to 1.
        seed (int, optional): random seed of dropout. Defaults to 2022.
        cos_sim (bool, optional): whether use cosine similarity or not. Defaults to False
        gamma (float, optional): smoothing factor for cosine similarity softmax. Defaults to 0.2.
    """
    if len(user_features) < 1:
        raise ValueError("Should have at least one user feature")
    if len(item_features) < 1:
        raise ValueError("Should have at least one item feature")

    # * Group features by their types
    u_dense_f, u_sparse_f, u_sparse_seq_f = split_features(user_features)
    i_dense_f, i_sparse_f, i_sparse_seq_f = split_features(item_features)
    if len(i_dense_f) > 0:
        warnings.WarningMessage(
            "DSSM doesn't support item dense feature now, they will be ignored!"
        )

    # * Get input and embedding layers
    input_layers = construct_input_layers(user_features + item_features)
    embd_layers = construct_embedding_layers(user_features + item_features, l2_emb)

    # * Get embedding: input layer -> embedding layer (-> pooling layer)
    u_embd_outputs = OrderedDict()
    for feat in u_sparse_f.keys():
        u_embd_outputs[feat] = embd_layers[feat](input_layers[feat])
    for feat in u_sparse_seq_f.values():
        sparse_emb = embd_layers[feat.sparse_feat.name]
        seq_input = input_layers[feat.name]
        u_embd_outputs[feat.name] = SequencePoolingLayer("mean")(sparse_emb(seq_input))

    # * Get full item embedding: input layer -> full value list -> embedding layer (-> pooling layer)
    i_embd_outputs = OrderedDict()
    for feat in i_sparse_f.keys():
        embd_layer_input = ValueTable(full_item_dict[feat])(input_layers[feat])
        i_embd_outputs[feat] = embd_layers[feat](embd_layer_input)
    for feat in i_sparse_seq_f.values():
        sparse_emb = embd_layers[feat.sparse_feat.name]
        seq_input = ValueTable(full_item_dict[feat.name])(input_layers[feat.name])
        i_embd_outputs[feat.name] = SequencePoolingLayer("mean")(sparse_emb(seq_input))
        i_embd_outputs[feat.name] = tf.squeeze(i_embd_outputs[feat.name])
        # * shape: (batch, 1, n) -> (batch, n)

    # * Concat input layers -> DNN
    u_dnn_input = concat_inputs(
        [input_layers[k] for k in u_dense_f.keys()], list(u_embd_outputs.values())
    )
    i_dnn_input = concat_inputs([], list(i_embd_outputs.values()))

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

    # * Sampled cosine similarity softmax output
    output = SampledSoftmaxLayer(num_sampled=num_sampled)(
        [
            tf.nn.l2_normalize(i_embedding) * gamma if cos_sim else i_embedding,
            tf.nn.l2_normalize(u_embedding) if cos_sim else u_embedding,
            input_layers[item_id_name],
        ]
    )

    # * Construct model
    def gather_embedding(inputs):
        full_item_embd, index = inputs
        return tf.squeeze(tf.gather(full_item_embd, index), axis=1)

    item_embedding = Lambda(gather_embedding)([i_embedding, input_layers[item_id_name]])

    user_inputs = [input_layers[f.name] for f in user_features]
    item_inputs = [input_layers[f.name] for f in item_features]
    model = Model(inputs=list(input_layers.values()), outputs=output)
    model.__setattr__("user_input", user_inputs)
    model.__setattr__("user_embedding", u_embedding)
    model.__setattr__("item_input", item_inputs)
    model.__setattr__("item_embedding", item_embedding)

    return model
