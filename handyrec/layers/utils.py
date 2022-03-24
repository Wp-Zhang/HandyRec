import collections
from typing import List, Union, OrderedDict
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Concatenate, Flatten
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from handyrec.features import SparseFeature, DenseFeature, SparseSeqFeature
from handyrec.features.utils import split_features
from handyrec.layers.tools import CustomEmbedding


def construct_input_layers(
    features: Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]
) -> OrderedDict[str, Input]:
    """Generate input layers repectively for each feature

    DenseFeature(FEAT_NAME) -> Input(FEAT_NAME)
    SparseFeature(FEAT_NAME) -> Input(FEAT_NAME)
    SparseSeqFeature(FEAT_NAME) -> Input(sparse_feat.FEAT_NAME) base sparse feature
                                   Input(FEAT_NAME)             sparse feature index seq
                                   Input(FEAT_NAME_len)         sparse feature seq length

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
        input_layers[feat.sparse_feat.name] = Input(
            shape=(1,),
            name=feat.sparse_feat.name,
            dtype=feat.sparse_feat.dtype,
        )
        input_layers[feat.name] = Input(
            shape=(feat.seq_len,), name=feat.name, dtype=feat.dtype
        )

    return input_layers


def construct_embedding_layers(
    sparse_features: Union[List[SparseFeature], List[SparseSeqFeature]], l2_reg: float
) -> OrderedDict[str, Embedding]:
    """Generate embedding layers for sparse features

    Args:
        sparse_features (Union[List[SparseFeature], List[SparseSeqFeature]]): sparse feature list
        l2_reg (float): l2 regularization parameter

    Returns:
        Dict[str, Embedding]: dictionary of embedding layers, {name: embedding layer}
    """
    embedding_layers = collections.OrderedDict()
    _, sparse_feats, sparse_seq_feats = split_features(sparse_features)

    for f in sparse_feats.values():
        embedding_layers[f.name] = CustomEmbedding(
            input_dim=f.vocab_size,
            output_dim=f.embdding_dim,
            embeddings_regularizer=l2(l2_reg),
            name="embed_" + f.name,
        )

    for f in sparse_seq_feats.values():
        embedding_layers[f.sparse_feat.name] = CustomEmbedding(
            input_dim=f.sparse_feat.vocab_size,
            output_dim=f.sparse_feat.embdding_dim,
            embeddings_regularizer=l2(l2_reg),
            name="embed_seq_" + f.sparse_feat.name,
            mask_zero=True,
        )

    return embedding_layers


def concatenate(inputs, axis: int = -1):
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


def concat_inputs(
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
        dense = concatenate(dense_inputs, axis)
        sparse = concatenate(embd_inputs, axis)
        if not keepdims:
            dense = Flatten()(dense)
            sparse = Flatten()(sparse)
        dense = tf.cast(dense, tf.float32)
        sparse = tf.cast(sparse, tf.float32)
        return concatenate([dense, sparse], axis)
    elif len(dense_inputs) > 0:
        output = concatenate(dense_inputs, axis)
        if not keepdims:
            output = Flatten()(output)
        return output
    elif len(embd_inputs) > 0:
        output = concatenate(embd_inputs, axis)
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
