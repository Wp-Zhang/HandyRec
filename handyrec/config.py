from typing import Dict, Union
import box
import numpy as np

from handyrec.features import (
    FeaturePool,
    FeatureGroup,
    EmbdFeatureGroup,
    DenseFeature,
    SparseFeature,
    SparseSeqFeature,
)


class ConfigLoader:
    def __init__(self, config: Union[str, Dict]):
        assert isinstance(config, (str, dict)), "config must be a string or a dict"
        if isinstance(config, str):
            self.config = box.box_from_file(config)
        else:
            self.config = box.Box(config)

    def _construct_dense_feat(self, name, feat_config, feature_dim):
        feat_config.name = feat_config.get("name", name)
        feat = DenseFeature(**feat_config)
        return feat

    def _construct_sparse_feat(self, name, feat_config, feature_dim):
        feat_config.name = feat_config.get("name", name)
        feat_config.vocab_size = feat_config.get("vocab_size", feature_dim[name])
        feat = SparseFeature(**feat_config)
        return feat

    def _construct_sparse_seq_feat(self, name, feat_config, feature_dim):
        unit_config = feat_config.unit
        unit_name = list(unit_config.keys())[0]
        unit_config = unit_config[unit_name]
        unit_config.name = unit_name
        unit = self._construct_sparse_feat(unit_name, unit_config, feature_dim)

        feat_config.name = feat_config.get("name", name)
        feat_config.unit = unit
        feat = SparseSeqFeature(**feat_config)
        return feat

    def get_feature_group(
        self,
        feature_group_name: str,
        feature_pool: FeaturePool,
        feature_dim: Dict = None,
        value_dict: Dict = None,
    ) -> Union[FeatureGroup, EmbdFeatureGroup]:
        fg_config = self.config.FeatureGroups[feature_group_name]

        dense_features = []
        for feat_name in fg_config.get("DenseFeatures", []):
            feat_config = fg_config.DenseFeatures[feat_name]
            feat = self._construct_dense_feat(feat_name, feat_config, feature_dim)
            dense_features.append(feat)

        sparse_features = []
        for feat_name in fg_config.get("SparseFeatures", []):
            feat_config = fg_config.SparseFeatures[feat_name]
            feat = self._construct_sparse_feat(feat_name, feat_config, feature_dim)
            sparse_features.append(feat)

        sparse_seq_features = []
        for feat_name in fg_config.get("SparseSeqFeatures", []):
            feat_config = fg_config.SparseSeqFeatures[feat_name]
            feat = self._construct_sparse_seq_feat(feat_name, feat_config, feature_dim)
            sparse_seq_features.append(feat)

        cfg = {
            "name": fg_config.name,
            "features": dense_features + sparse_features + sparse_seq_features,
            "feature_pool": feature_pool,
        }
        if fg_config.get("l2_embd", None):
            cfg["l2_embd"] = float(fg_config.l2_embd)

        if fg_config["type"] == "EmbdFeatureGroup":
            cfg["id_name"] = fg_config.id_name
            cfg["embd_dim"] = fg_config.get("embd_dim", None)
            cfg["value_dict"] = value_dict
            if fg_config.get("pool_method", None):
                cfg["pool_method"] = fg_config.pool_method
            feature_group = EmbdFeatureGroup(**cfg)
        else:
            feature_group = FeatureGroup(**cfg)

        return feature_group

    def prepare_features(
        self,
        feature_dim: Dict = None,
        data: Dict = None,
        pretrained_embd: Dict = None,
    ) -> Dict:
        feature_pool = FeaturePool(pretrained_embd)
        result = {"feature_pool": feature_pool}

        for fg in self.config.FeatureGroups:
            fg_cfg = self.config.FeatureGroups[fg]
            if fg_cfg.type == "EmbdFeatureGroup":
                group_feats = [x for x in fg_cfg.get("DenseFeatures", [])]
                group_feats += [x for x in fg_cfg.get("SparseFeatures", [])]
                group_feats += [x for x in fg_cfg.get("SparseSeqFeatures", [])]
                value_dict = {
                    f: np.array(data["item"][f].tolist()) for f in group_feats
                }
                feature_group = self.get_feature_group(
                    fg, feature_pool, feature_dim, value_dict
                )
                result["value_dict"] = value_dict
            else:
                feature_group = self.get_feature_group(fg, feature_pool, feature_dim)
            result[fg] = feature_group

        return result
