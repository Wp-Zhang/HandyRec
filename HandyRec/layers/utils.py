from tensorflow.keras.layers import Embedding
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2
from ..features import SparseFeature, DenseFeature, SparseSeqFeature
from ..features.utils import split_features
from typing import List, Union, OrderedDict
import collections


def get_input_layers(
    features: Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]
) -> OrderedDict[str, Input]:
    """Generate input layers repectively for each feature

    Args:
        features (Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]): feature list

    Returns:
        Dict[str, InputLayer]: dictionary of input layers, {name: input layer}
    """
    input_layers = collections.OrderedDict()
    dense_feats, sparse_feats, sparse_seq_feats = split_features(features)

    for feat in dense_feats:
        input_layers[feat.name] = Input(shape=(1,), name=feat.name, dtype=feat.dtype)

    for feat in sparse_feats:
        input_layers[feat.name] = Input(shape=(1,), name=feat.name, dtype=feat.dtype)

    for feat in sparse_seq_feats:
        input_layers[feat.sparse_feat.name] = Input(
            shape=(feat.seq_len,),
            name=feat.sparse_feat.name,
            dtype=feat.sparse_feat.dtype,
        )
        input_layers[feat.name] = Input(shape=(1,), name=feat.name, dtype=feat.dtype)

    return input_layers


def get_embedding_layers(
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

    for f in sparse_feats:
        embedding_layers[f.name] = Embedding(
            input_dim=f.vocab_size,
            output_dim=f.embdding_dim,
            embeddings_regularizer=l2(l2_reg),
            name="embed_" + f.name,
        )

    for f in sparse_seq_feats:
        embedding_layers[f.sparse_feat.name] = Embedding(
            input_dim=f.sparse_feat.vocab_size,
            output_dim=f.sparse_feat.embdding_dim,
            embeddings_regularizer=l2(l2_reg),
            name="embed_seq_" + f.sparse_feat.name,
            mask_zero=True,
        )

    return embedding_layers
