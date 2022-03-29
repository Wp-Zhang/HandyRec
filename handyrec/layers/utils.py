import collections
from typing import List, Union, OrderedDict, Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from handyrec.features import SparseFeature, DenseFeature, SparseSeqFeature
from handyrec.features.utils import split_features
from .tools import CustomEmbedding


def construct_input_layers(
    features: Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]
) -> OrderedDict[str, Input]:
    """Construct input layers repectively for each feature

    Args:
        features (Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]): feature list

    Returns:
        Dict[str, InputLayer]: dictionary of input layers, {name: input layer}
    """
    input_layers = collections.OrderedDict()
    dense_feats, sparse_feats, sparse_seq_feats = split_features(features)

    for feat in dense_feats.values():
        input_layers[feat.name] = Input(shape=(1,), name=feat.name, dtype=feat.dtype)

    for feat in sparse_feats.values():
        input_layers[feat.name] = Input(shape=(1,), name=feat.name, dtype=feat.dtype)

    for feat in sparse_seq_feats.values():
        input_layers[feat.name] = Input(
            shape=(feat.seq_len,), name=feat.name, dtype=feat.dtype
        )  # * sparse feature index seq

    return input_layers


def construct_embedding_layers(
    sparse_features: Union[List[SparseFeature], List[SparseSeqFeature]],
    l2_reg: float,
    pretrained_embeddings: Dict[str, Union[np.ndarray, tf.Tensor]] = None,
) -> OrderedDict[str, CustomEmbedding]:
    """Construct embedding layers for sparse features

    Args:
        sparse_features (Union[List[SparseFeature], List[SparseSeqFeature]]): sparse feature list
        l2_reg (float): l2 regularization parameter
        pretrained_embeddings (Dict[str, Union[np.ndarray, tf.Tensor]], optional):
            pretrained embedding dict {name: weights}. Defaults to None

    Returns:
        Dict[str, CustomEmbedding]: dictionary of embedding layers, {name: embedding layer}
    """
    embedding_layers = collections.OrderedDict()
    _, sparse_feats, sparse_seq_feats = split_features(sparse_features)

    sparse_feats = list(sparse_feats.values())
    sparse_seq_feats = [seq_f.sparse_feat for seq_f in sparse_seq_feats.values()]

    for feat in sparse_feats + sparse_seq_feats:
        embedding_layers[feat.name] = CustomEmbedding(
            input_dim=feat.vocab_size,
            output_dim=feat.embdding_dim,
            embeddings_regularizer=l2(l2_reg),
            trainable=feat.trainable,
            name="embed_" + feat.name,
            mask_zero=feat in sparse_seq_feats,
        )

        # * Load pretrained embedding
        if pretrained_embeddings and (feat.name in pretrained_embeddings.keys()):
            embedding_layers[feat.name].set_weights(pretrained_embeddings[feat.name])

    return embedding_layers


def _concat(inputs, axis: int = -1):
    """Concatenate list of input, handle the case when len(inputs)=1

    Args:
        inputs : list of input
        axis (int, optional): concatenate axis. Defaults to -1.
        # mask (bool, optional): whether to keep masks of input tensors. Defaults to Ture.

    Returns:
        _type_: concatenated input
    """
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def concat(
    dense_inputs: List, embd_inputs: List, axis: int = -1, keepdims: bool = False
):
    """Concatenate dense features and embedding of sparse features together

    Args:
        dense_inputs (List): dense features
        embd_inputs (List): embedding of sparse features
        axis (int, optional): concatenate axis. Deafults to `-1`
        keepdims (bool, optional): whether to flatten all inputs before concatenating. Defaults to `False`
        # mask (bool, optional): whether to keep masks of input tensors. Defaults to Ture.
    """
    if len(dense_inputs) + len(embd_inputs) == 0:
        raise ValueError("Number of inputs should be larger than 0")

    if len(dense_inputs) > 0 and len(embd_inputs) > 0:
        dense = _concat(dense_inputs, axis)
        sparse = _concat(embd_inputs, axis)
        if not keepdims:
            dense = Flatten()(dense)
            sparse = Flatten()(sparse)

        # * Change dtype
        if dense.dtype != sparse.dtype:
            if dense.dtype.is_integer:
                dense = tf.cast(dense, sparse.dtype)
            else:
                sparse = tf.cast(sparse, dense.dtype)

        return _concat([dense, sparse], axis)

    if len(dense_inputs) > 0:
        output = _concat(dense_inputs, axis)
        if not keepdims:
            output = Flatten()(output)
        return output

    if len(embd_inputs) > 0:
        output = _concat(embd_inputs, axis)
        if not keepdims:
            output = Flatten()(output)
        return output


def sampledsoftmaxloss(y_true, y_pred):
    """Helper function for calculating sampled softmax loss

    Args:
        y_true : label
        y_pred : prediction

    Returns:
        _type_: loss
    """
    return tf.reduce_mean(y_pred)
