from tests.test_helper import get_pairwise_dataset
from handyrec.models.ranking import FMLPRec
from handyrec.features import (
    SparseFeature,
    SparseSeqFeature,
    FeatureGroup,
    FeaturePool,
)

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import pytest


def test_FMLPRec():
    dataset = get_pairwise_dataset()

    user_features = ["user_id"]
    item_features = ["movie_id"]
    inter_features = ["hist_movie"]
    train_data, valid_data, test_data, test_label = dataset.load_dataset(
        user_features, item_features, inter_features, 5
    )
    feature_dim = dataset.get_feature_dim(user_features, item_features, [])

    feat_pool = FeaturePool()

    rank_item_seq_features = [
        SparseSeqFeature(
            SparseFeature("movie_id", feature_dim["movie_id"], 8), "hist_movie", 2
        )
    ]
    item_seq_feat_group = FeatureGroup("item_seq", rank_item_seq_features, feat_pool)

    # * --------------------------------------------------------------------------------

    with pytest.raises(AttributeError):
        feat_pool2 = FeaturePool()
        rank_item_seq_features2 = [
            SparseSeqFeature(
                SparseFeature("movie_id", feature_dim["movie_id"], 8), "hist_movie", 3
            )
        ]
        item_seq_feat_group2 = FeatureGroup(
            "item_seq", rank_item_seq_features2, feat_pool2
        )

        rank_model = FMLPRec(item_seq_feat_group2, dropout=0.5, block_num=3)
    # * --------------------------------------------------------------------------------

    rank_model = FMLPRec(item_seq_feat_group, dropout=0.5, block_num=3)

    rank_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=binary_crossentropy
    )
    history = rank_model.fit(x=train_data, validation_data=valid_data, epochs=2)

    model = tf.keras.Model(
        inputs=rank_model.real_inputs, outputs=rank_model.real_outputs
    )
    pred = model.predict(test_data, batch_size=5)
