from .features import DenseFeature, SparseFeature, SparseSeqFeature
from typing import Union, List, Tuple, OrderedDict
import collections


def split_features(
    features: Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]
) -> Tuple[OrderedDict]:
    """Group feature list into different types of feature lists

    Args:
        features (Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]): feature list

    Returns:
        Tuple[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]: three types of feature list
    """
    dense_feats = collections.OrderedDict()
    sparse_feats = collections.OrderedDict()
    sparse_seq_feats = collections.OrderedDict()

    for feat in features:
        if isinstance(feat, DenseFeature):
            dense_feats[feat.name] = feat
        elif isinstance(feat, SparseFeature):
            sparse_feats[feat.name] = feat
        elif isinstance(feat, SparseSeqFeature):
            sparse_seq_feats[feat.name] = feat

    return dense_feats, sparse_feats, sparse_seq_feats
