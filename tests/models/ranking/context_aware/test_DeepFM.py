from tests.test_helper import get_pointwise_dataset
from handyrec.models.ranking import DeepFM
from handyrec.config import ConfigLoader
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
    feature_dim["genre_id"] = 19

    cfg = ConfigLoader("tests/ml-1m-test/DeepFM_cfg.yaml")
    feature_groups = cfg.prepare_features(feature_dim)
    fm_feature_group = feature_groups["fm_feature_group"]
    dnn_feature_group = feature_groups["dnn_feature_group"]

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
