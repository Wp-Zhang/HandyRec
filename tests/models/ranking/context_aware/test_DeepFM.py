from tests.test_helper import get_pointwise_dataset
from handyrec.models.ranking import DeepFM
from handyrec.features import (
    DenseFeature,
    SparseFeature,
    SparseSeqFeature,
    FeatureGroup,
    FeaturePool,
)
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import pytest


def test_DeepFM():
    dataset = get_pointwise_dataset(task="ranking")

    user_features = ["user_id", "gender", "occupation", "zip", "age"]
    item_features = ["movie_id", "year", "genres"]
    inter_features = ["hist_movie"]
    train_data, valid_data, test_data, test_label = dataset.load_dataset(
        user_features, item_features, inter_features, 5
    )
    feature_dim = dataset.get_feature_dim(user_features, item_features, [])

    feat_pool = FeaturePool()

    rank_fm_features = [
        *[SparseFeature(x, feature_dim[x], 8) for x in user_features],
        SparseSeqFeature(
            SparseFeature("movie_id", feature_dim["movie_id"], 8), "hist_movie", 2
        ),
        *[SparseFeature(x, feature_dim[x], 8) for x in item_features[:1]],
        *[SparseSeqFeature(SparseFeature("genre_id", 19, 8), "genres", 3)],
        DenseFeature("year", dtype="int32"),
    ]
    fm_feature_group = FeatureGroup("FM", rank_fm_features, feat_pool)
    rank_dnn_feats = rank_fm_features
    dnn_feature_group = FeatureGroup("DNN", rank_dnn_feats, feat_pool)

    # * ----------------------------------------------------------------------

    with pytest.raises(ValueError) as e_info:
        rank_model = DeepFM(
            fm_feature_group,
            dnn_feature_group,
            dnn_hidden_units=(8, 4),
            dnn_dropout=0.2,
            l2_dnn=0.2,
            dnn_bn=True,
        )
    assert e_info.type is ValueError

    # * ----------------------------------------------------------------------

    rank_model = DeepFM(
        fm_feature_group,
        dnn_feature_group,
        dnn_hidden_units=(8, 1),
        dnn_dropout=0.2,
        l2_dnn=0.2,
        dnn_bn=True,
    )

    rank_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=binary_crossentropy
    )
    history = rank_model.fit(x=train_data, validation_data=valid_data, epochs=2)
    pred = rank_model.predict(test_data, batch_size=5)
