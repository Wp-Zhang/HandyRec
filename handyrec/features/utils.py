from typing import Union, List, Tuple, OrderedDict
from .type import DenseFeature, SparseFeature, SparseSeqFeature


def split_features(
    features: Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]
) -> Tuple[
    OrderedDict[str, DenseFeature],
    OrderedDict[str, SparseFeature],
    OrderedDict[str, SparseSeqFeature],
]:
    """Group feature list into different types of feature lists

    Args:
        features (Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]): feature list

    Returns:
        Tuple[OrderedDict[str,DenseFeature], OrderedDict[str,SparseFeature], OrderedDict[str,SparseSeqFeature]]: three types of feature list
    """
    dense_feats = OrderedDict()
    sparse_feats = OrderedDict()
    sparse_seq_feats = OrderedDict()

    for feat in features:
        if isinstance(feat, DenseFeature):
            dense_feats[feat.name] = feat
        elif isinstance(feat, SparseFeature):
            sparse_feats[feat.name] = feat
        elif isinstance(feat, SparseSeqFeature):
            sparse_seq_feats[feat.name] = feat

    return dense_feats, sparse_feats, sparse_seq_feats
