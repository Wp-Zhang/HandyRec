from .features import DenseFeature, SparseFeature, SparseSeqFeature
from typing import Union, List, Tuple


def split_features(
    features: Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]
) -> Tuple[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]:
    """Group feature list into different types of feature lists

    Args:
        features (Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]): feature list

    Returns:
        Tuple[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]: three types of feature list
    """
    dense_feats = [f for f in features if isinstance(f, DenseFeature)]
    sparse_feats = [f for f in features if isinstance(f, SparseFeature)]
    sparse_seq_feats = [f for f in features if isinstance(f, SparseSeqFeature)]

    return dense_feats, sparse_feats, sparse_seq_feats
