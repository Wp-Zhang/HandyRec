from handyrec.features import DenseFeature, SparseFeature, SparseSeqFeature
from handyrec.features.utils import split_features


def test_split_features_func():
    dense_feats = [DenseFeature("d1", dim=1), DenseFeature("d2", dim=2)]
    sparse_feats = [
        SparseFeature("s1", vocab_size=10, embedding_dim=16),
        SparseFeature("s2", vocab_size=20, embedding_dim=16),
    ]
    sparse_seq_feats = [SparseSeqFeature(sparse_feats[0], "s1_seq", seq_len=15)]

    dense_feats_dict, sparse_feats_dict, sparse_seq_feats_dict = split_features(
        dense_feats + sparse_feats + sparse_seq_feats
    )

    assert dense_feats_dict["d1"] == dense_feats[0]
    assert dense_feats_dict["d2"] == dense_feats[1]
    assert sparse_feats_dict["s1"] == sparse_feats[0]
    assert sparse_feats_dict["s2"] == sparse_feats[1]
    assert sparse_seq_feats_dict["s1_seq"] == sparse_seq_feats[0]
