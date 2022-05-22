from tests.test_helper import get_pointwise_dataset
from handyrec.models.ranking import DIN
from handyrec.features import (
    SparseFeature,
    SparseSeqFeature,
    FeatureGroup,
    FeaturePool,
)

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy


def test_DIN():
    dataset = get_pointwise_dataset(task="ranking")

    user_features = ["user_id", "gender", "occupation", "zip", "age"]
    item_features = ["movie_id", "year", "genres"]
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
    rank_other_feats = [
        SparseFeature(x, feature_dim[x], 8) for x in user_features + item_features[:-1]
    ] + [SparseSeqFeature(SparseFeature("genre_id", 19, 8), "genres", 3)]
    other_feature_group = FeatureGroup("user", rank_other_feats, feat_pool)

    rank_model = DIN(
        item_seq_feat_group,
        other_feature_group,
        dnn_hidden_units=(8, 1),
        dnn_activation="dice",
        dnn_dropout=0.2,
        l2_dnn=0.2,
        dnn_bn=True,
        lau_dnn_hidden_units=(4, 1),
        lau_dnn_activation="dice",
        lau_dnn_dropout=0.0,
        lau_l2_dnn=0.2,
        lau_dnn_bn=False,
    )

    rank_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=binary_crossentropy
    )
    history = rank_model.fit(x=train_data, validation_data=valid_data, epochs=2)
    pred = rank_model.predict(test_data, batch_size=5)
