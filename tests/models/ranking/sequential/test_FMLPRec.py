from tests.test_helper import get_pairwise_dataset
from handyrec.models.ranking import FMLPRec
from handyrec.config import ConfigLoader

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

    # * --------------------------------------------------------------------------------

    with pytest.raises(AttributeError):
        cfg_dict = {
            "FeatureGroups": {
                "item_seq_feat_group": {
                    "type": "FeatureGroup",
                    "name": "item_seq",
                    "SparseSeqFeatures": {
                        "hist_movie": {
                            "unit": {"movie_id": {"embedding_dim": 8}},
                            "seq_len": 3,
                        }
                    },
                }
            }
        }

        cfg = ConfigLoader(cfg_dict)
        feature_groups = cfg.prepare_features(feature_dim)
        item_seq_feat_group = feature_groups["item_seq_feat_group"]

        rank_model = FMLPRec(item_seq_feat_group, dropout=0.5, block_num=3)
    # * --------------------------------------------------------------------------------

    cfg = ConfigLoader("tests/ml-1m-test/FMLPRec_cfg.yaml")
    feature_groups = cfg.prepare_features(feature_dim)
    item_seq_feat_group = feature_groups["item_seq_feat_group"]

    rank_model = FMLPRec(item_seq_feat_group, dropout=0.5, block_num=3)

    rank_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=binary_crossentropy
    )
    history = rank_model.fit(x=train_data, validation_data=valid_data, epochs=2)

    model = tf.keras.Model(
        inputs=rank_model.real_inputs, outputs=rank_model.real_outputs
    )
    pred = model.predict(test_data, batch_size=5)
