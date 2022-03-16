import tensorflow as tf
from tensorflow.keras.layers import Embedding, Concatenate, Flatten
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from ..features import SparseFeature, DenseFeature, SparseSeqFeature
from ..features.utils import split_features
from typing import List, Union, OrderedDict
import collections


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
        input_layers[feat.name + "_len"] = Input(
            shape=(1,), name=feat.name + "_len", dtype=feat.dtype
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
        embedding_layers[f.name] = Embedding(
            input_dim=f.vocab_size,
            output_dim=f.embdding_dim,
            embeddings_regularizer=l2(l2_reg),
            name="embed_" + f.name,
        )

    for f in sparse_seq_feats.values():
        embedding_layers[f.sparse_feat.name] = Embedding(
            input_dim=f.sparse_feat.vocab_size,
            output_dim=f.sparse_feat.embdding_dim,
            embeddings_regularizer=l2(l2_reg),
            name="embed_seq_" + f.sparse_feat.name,
            mask_zero=True,
        )

    return embedding_layers


def concatenate(inputs, axis: int = -1):
    """Concatenate list of input, handle the case of len(inputs)=1

    Args:
        inputs : list of input
        axis (int, optional): concatenate axis. Defaults to -1.

    Returns:
        _type_: concatenated input
    """
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def concat_inputs(dense_inputs: List, embd_inputs: List):
    """Concatenate dense features and embedding of sparse features together

    Args:
        dense_inputs (List): dense features
        embd_inputs (List): embedding of sparse features
    """
    if len(dense_inputs) + len(embd_inputs) == 0:
        raise ValueError("Number of inputs should be larger than 0")

    if len(dense_inputs) > 0 and len(embd_inputs) > 0:
        dense = Flatten()(concatenate(dense_inputs))
        sparse = Flatten()(concatenate(embd_inputs))
        dense = tf.cast(dense, tf.float32)
        sparse = tf.cast(sparse, tf.float32)
        return concatenate([dense, sparse])
    elif len(dense_inputs) > 0:
        return Flatten()(concatenate(dense_inputs))
    elif len(embd_inputs) > 0:
        return Flatten()(concatenate(embd_inputs))


def sampledsoftmaxloss(y_true, y_pred):
    return tf.reduce_mean(y_pred)
