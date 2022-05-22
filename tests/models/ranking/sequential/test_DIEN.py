from tests.test_helper import get_sequencewise_dataset
from handyrec.models.ranking import DIEN
from handyrec.features import (
    SparseFeature,
    SparseSeqFeature,
    FeatureGroup,
    FeaturePool,
)

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import pytest


def test_DIEN():
    dataset = get_sequencewise_dataset()

    user_features = ["user_id", "gender", "occupation", "zip", "age"]
    item_features = ["movie_id", "year", "genres"]
    inter_features = ["hist_movie"]
    train_data, valid_data, test_data, test_label = dataset.load_dataset(
        user_features, item_features, inter_features, 5
    )
    feature_dim = dataset.get_feature_dim(user_features, item_features, [])

    feat_pool = FeaturePool()

    item_seq_features = [
        SparseSeqFeature(
            SparseFeature("movie_id", feature_dim["movie_id"], 8), "hist_movie", 2
        )
    ]
    neg_item_seq_features = [
        SparseSeqFeature(
            SparseFeature("movie_id", feature_dim["movie_id"], 8), "neg_hist_movie", 2
        )
    ]
    item_seq_feat_group = FeatureGroup("item_seq", item_seq_features, feat_pool)
    neg_item_seq_feat_group = FeatureGroup(
        "neg_item_seq", neg_item_seq_features, feat_pool
    )
    rank_other_feats = [
        SparseFeature(x, feature_dim[x], 8) for x in user_features + item_features[:-1]
    ] + [SparseSeqFeature(SparseFeature("genre_id", 19, 8), "genres", 3)]
    other_feature_group = FeatureGroup("user", rank_other_feats, feat_pool)

    # * --------------------------------------------------------------------------------

    with pytest.raises(ValueError):
        neg_item_seq_feat_group2 = FeatureGroup("neg_item_seq", [], feat_pool)
        rank_model = DIEN(
            item_seq_feat_group,
            neg_item_seq_feat_group2,
            other_feature_group,
            gru_dropout=0.1,
            lau_dnn_hidden_units=(2, 1),
            lau_dnn_activation="dice",
            lau_dnn_dropout=0.0,
            lau_l2_dnn=0.2,
            lau_dnn_bn=False,
            augru_units=8,
            dnn_hidden_units=(16, 1),
            dnn_activation="dice",
            dnn_dropout=0.2,
            l2_dnn=0.2,
            dnn_bn=True,
        )
    # * --------------------------------------------------------------------------------

    rank_model = DIEN(
        item_seq_feat_group,
        neg_item_seq_feat_group,
        other_feature_group,
        gru_dropout=0.1,
        lau_dnn_hidden_units=(2, 1),
        lau_dnn_activation="dice",
        lau_dnn_dropout=0.0,
        lau_l2_dnn=0.2,
        lau_dnn_bn=False,
        augru_units=8,
        dnn_hidden_units=(16, 1),
        dnn_activation="dice",
        dnn_dropout=0.2,
        l2_dnn=0.2,
        dnn_bn=True,
    )

    rank_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=binary_crossentropy
    )
    history = rank_model.fit(x=train_data, validation_data=valid_data, epochs=2)

    model = tf.keras.Model(inputs=rank_model.real_inputs, outputs=rank_model.outputs)
    pred = model.predict(test_data, batch_size=5)
