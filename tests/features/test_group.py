from handyrec.features import DenseFeature, SparseFeature, SparseSeqFeature
from handyrec.features import FeaturePool, FeatureGroup, EmbdFeatureGroup
import tensorflow as tf
import numpy as np
import pytest


def test_FeatureGroup():
    dense_feats = [DenseFeature("d1", dim=1), DenseFeature("d2", dim=2)]
    sparse_feats = [
        SparseFeature("s1", vocab_size=10, embedding_dim=16),
        SparseFeature("s2", vocab_size=20, embedding_dim=16),
    ]
    sparse_seq_feats = [
        SparseSeqFeature(
            SparseFeature("s3", vocab_size=24, embedding_dim=16), "s1_seq", seq_len=15
        )
    ]

    pool = FeaturePool({"s1": np.random.rand(10, 16)})
    fg = FeatureGroup("FG", dense_feats + sparse_feats + sparse_seq_feats, pool)

    dense_outputs, embd_outputs = fg.embedding_lookup()

    assert dense_outputs[0].shape[1:] == (1)
    assert dense_outputs[1].shape[1:] == (2)

    for embd_out in embd_outputs:
        assert embd_out.shape[1:] == (1, 16)


def test_EmbdFeatureGroup():
    dense_feats = [DenseFeature("d1", dim=1), DenseFeature("d2", dim=2)]
    sparse_feats = [
        SparseFeature("s1", vocab_size=10, embedding_dim=16),
        SparseFeature("s2", vocab_size=20, embedding_dim=16),
    ]
    sparse_seq_feats = [SparseSeqFeature(sparse_feats[0], "s2_seq", seq_len=4)]

    value_dict = {
        "s1": [1, 2, 3],
        "s2": [11, 12, 13],
        "d1": [-1, -2, -3],
        "d2": [[1, 2], [3, 4], [5, 6]],
        "s2_seq": [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    }

    pool = FeaturePool()

    # * ------------------------------------------------------------------------

    with pytest.raises(ValueError) as e_info:
        fg = EmbdFeatureGroup(
            "FG",
            "XXX",
            dense_feats + sparse_feats + sparse_seq_feats,
            pool,
            value_dict,
            embd_dim=8,
        )
    assert e_info.type is ValueError

    with pytest.raises(ValueError) as e_info:
        sparse_seq_feats2 = [SparseSeqFeature(dense_feats[0], "d1_seq", seq_len=4)]
        fg = EmbdFeatureGroup(
            "FG",
            "s1",
            dense_feats + sparse_feats + sparse_seq_feats2,
            pool,
            value_dict,
            embd_dim=8,
        )
    assert e_info.type is ValueError

    # * ------------------------------------------------------------------------

    fg = EmbdFeatureGroup(
        "FG",
        "s1",
        dense_feats + sparse_feats + sparse_seq_feats,
        pool,
        value_dict,
        embd_dim=8,
    )

    item_id = fg.id_input
    embd_outputs1 = fg.get_embd(item_id, compress=False)
    assert embd_outputs1.shape == (3, 3 * 16 + 1 + 2)

    embd_outputs2 = fg.get_embd(item_id, compress=True)
    assert embd_outputs2.shape == (3, 8)

    seq_id_input = tf.constant([[1, 2, 3], [3, 2, 1]])

    lookup_outputs = fg.lookup(seq_id_input, compress=False)
    assert lookup_outputs.shape == (2, 3, 3 * 16 + 1 + 2)

    lookup_outputs2 = fg.lookup(item_id, compress=False)
    assert (
        lookup_outputs2.shape[0] is None and lookup_outputs2.shape[1] == 3 * 16 + 1 + 2
    )

    call_outputs, call_mask = fg(seq_id_input)
    assert call_outputs.shape == (2, 3, 8)
    assert call_mask.shape == (2, 3, 8)


def test_FeaturePool_init_input():
    pool = FeaturePool()

    params = {"name": "d1", "shape": (1,), "dtype": "float32"}
    pool.init_input("d1", params)

    with pytest.raises(AttributeError) as e_info:
        params = {"name": "d1", "shape": (2,), "dtype": "float32"}
        pool.init_input("d1", params)
    assert e_info.type is AttributeError

    with pytest.raises(AttributeError) as e_info:
        params = {"name": "d1", "shape": (1,), "dtype": "int32"}
        pool.init_input("d1", params)
    assert e_info.type is AttributeError


def test_FeaturePool_init_embd():
    pool = FeaturePool()

    params = {
        "name": "embd_s1",
        "input_dim": 12,
        "output_dim": 128,
        "trainable": True,
        "weights": None,
        "mask_zero": False,
    }
    pool.init_embd("s1", params)

    params["mask_zero"] = True
    pool.init_embd("s1", params)

    with pytest.raises(AttributeError) as e_info:
        params["input_dim"] = 24
        pool.init_embd("s1", params)
    assert e_info.type is AttributeError

    with pytest.raises(AttributeError) as e_info:
        params["output_dim"] = 64
        pool.init_embd("s1", params)
    assert e_info.type is AttributeError


def test_FeaturePool_init_pool():
    pool = FeaturePool()

    pool.init_pool("POOL_1", {"name": "POOL_1", "method": "mean"})
    with pytest.raises(AttributeError) as e_info:
        pool.init_pool("POOL_1", {"name": "POOL_1", "method": "max"})
    assert e_info.type is AttributeError
