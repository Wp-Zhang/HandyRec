from tests.test_helper import get_ml_test_data, get_pointwise_dataset
from handyrec.models.retrieval import DSSM
from handyrec.layers.utils import sampledsoftmaxloss
from handyrec.features import (
    SparseFeature,
    SparseSeqFeature,
    FeatureGroup,
    EmbdFeatureGroup,
    FeaturePool,
)
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
import pytest


def test_DSSM():
    data = get_ml_test_data()
    dataset = get_pointwise_dataset()

    user_features = ["user_id", "gender", "occupation"]
    item_features = ["movie_id", "genres"]
    inter_features = ["hist_movie"]
    train_data, valid_data, test_data, test_label = dataset.load_dataset(
        user_features, item_features, inter_features, 5
    )
    feature_dim = dataset.get_feature_dim(user_features, item_features, [])

    feat_pool1 = FeaturePool()
    all_item_model_input = {
        f: np.array(data["item"][f].tolist()) for f in item_features
    }

    retrieve_item_features = [
        SparseFeature("movie_id", feature_dim["movie_id"], 8),
        SparseSeqFeature(SparseFeature("genre_id", 19, 8), "genres", 3),
    ]
    item_feature_group = EmbdFeatureGroup(
        name="item",
        id_name="movie_id",
        features=retrieve_item_features,
        feature_pool=feat_pool1,
        value_dict=all_item_model_input,
        embd_dim=8,
    )

    retrieve_user_features = [
        *[SparseFeature(x, feature_dim[x], 8) for x in user_features],
        SparseSeqFeature(
            SparseFeature("movie_id", feature_dim["movie_id"], 8), "hist_movie", 2
        ),
    ]
    user_feature_group = FeatureGroup("user", retrieve_user_features, feat_pool1)

    # * --------------------------------------------------

    with pytest.raises(ValueError) as e_info:
        retrieve_model = DSSM(
            user_feature_group,
            user_feature_group,
            user_dnn_hidden_units=(16, 8),
            item_dnn_hidden_units=(16, 8),
            dnn_dropout=0.1,
            dnn_bn=True,
            num_sampled=1,
        )
    assert e_info.type == ValueError

    # * --------------------------------------------------

    retrieve_model = DSSM(
        user_feature_group,
        item_feature_group,
        user_dnn_hidden_units=(16, 8),
        item_dnn_hidden_units=(16, 8),
        dnn_dropout=0.1,
        dnn_bn=True,
        num_sampled=1,
    )

    retrieve_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=5e-4), loss=sampledsoftmaxloss
    )
    history = retrieve_model.fit(x=train_data, validation_data=valid_data, epochs=2)
    user_embedding_model = Model(
        inputs=retrieve_model.user_input, outputs=retrieve_model.user_embedding
    )
    item_embedding_model = Model(
        inputs=retrieve_model.item_input, outputs=retrieve_model.item_embedding
    )

    user_embs = user_embedding_model.predict(test_data, batch_size=5)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=5)

    assert user_embs.shape == (test_data["user_id"].shape[0], 8)
    assert item_embs.shape == (all_item_model_input["movie_id"].shape[0], 8)
