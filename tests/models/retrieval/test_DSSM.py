from tests.test_helper import get_ml_test_data, get_pointwise_dataset
from handyrec.models.retrieval import DSSM
from handyrec.layers.utils import sampledsoftmaxloss
from handyrec.config import ConfigLoader
import tensorflow as tf
from tensorflow.keras import Model
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
    feature_dim["genre_id"] = 19

    cfg = ConfigLoader("tests/ml-1m-test/DSSM_cfg.yaml")
    feature_groups = cfg.prepare_features(feature_dim, data)
    full_item_model_input = feature_groups["value_dict"]
    user_feature_group = feature_groups["user_feature_group"]
    item_feature_group = feature_groups["item_feature_group"]

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
    item_embs = item_embedding_model.predict(full_item_model_input, batch_size=5)

    assert user_embs.shape == (test_data["user_id"].shape[0], 8)
    assert item_embs.shape == (full_item_model_input["movie_id"].shape[0], 8)
