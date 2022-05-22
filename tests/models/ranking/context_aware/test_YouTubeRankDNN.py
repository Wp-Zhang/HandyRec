from tests.test_helper import get_pointwise_dataset
from handyrec.models.ranking import YouTubeRankDNN
from handyrec.features import (
    SparseFeature,
    SparseSeqFeature,
    FeatureGroup,
    FeaturePool,
)
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy


def test_YouTubeRankDNN():
    dataset = get_pointwise_dataset(task="ranking")

    user_features = ["user_id", "gender", "occupation"]
    item_features = ["movie_id", "genres"]
    inter_features = ["hist_movie"]
    train_data, valid_data, test_data, test_label = dataset.load_dataset(
        user_features, item_features, inter_features, 5
    )
    feature_dim = dataset.get_feature_dim(user_features, item_features, [])

    feat_pool = FeaturePool()

    rank_user_features = [
        *[SparseFeature(x, feature_dim[x], 8) for x in user_features],
        SparseSeqFeature(
            SparseFeature("movie_id", feature_dim["movie_id"], 8), "hist_movie", 2
        ),
    ]
    user_feature_group = FeatureGroup("user", rank_user_features, feat_pool)
    rank_item_feats = [
        *[SparseFeature(x, feature_dim[x], 8) for x in item_features[:-1]],
        SparseSeqFeature(SparseFeature("genre_id", 19, 8), "genres", 3),
    ]
    item_feature_group = FeatureGroup("item", rank_item_feats, feat_pool)

    rank_model = YouTubeRankDNN(
        user_feature_group,
        item_feature_group,
        dnn_hidden_units=(16, 8),
        dnn_dropout=0.2,
    )

    rank_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=binary_crossentropy
    )
    history = rank_model.fit(x=train_data, validation_data=valid_data, epochs=2)
    pred = rank_model.predict(test_data, batch_size=5)
