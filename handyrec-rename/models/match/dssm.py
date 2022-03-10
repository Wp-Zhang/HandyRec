from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from ...features import SparseFeature, DenseFeature, SparseSeqFeature
from ...features.utils import split_features
from ...layers import SparseSeqInput, SequencePoolingLayer
from ...layers.utils import construct_input_layers


class DSSM(Model):
    """DSSM model"""

    def __init__(
        self,
        user_features,
        item_features,
        num_sampled=1,
        user_dnn_hidden_units=(64, 32),
        item_dnn_hidden_units=(64, 32),
        dnn_activation="relu",
        l2_reg_embedding=1e-6,
        dnn_dropout=0,
    ):
        super(DSSM, self)._init__()

        user_dense, user_sparse, user_sparse_seq = split_features(user_features)
        item_dense, item_sparse, item_sparse_seq = split_features(item_features)

        self.user_feat_list = (
            [feat.name for feat in user_dense]
            + [feat.name for feat in user_sparse]
            + [feat.name for feat in user_sparse_seq]
        )

        self.item_feat_list = (
            [feat.name for feat in item_dense]
            + [feat.name for feat in item_sparse]
            + [feat.name for feat in item_sparse_seq]
        )

        assert len(self.user_feat_list) > 0, "Should have at least one user feature"
        assert len(self.item_feat_list) > 0, "Should have at least one item feature"

        user_input_layers = construct_input_layers(user_features)
        item_input_layers = construct_input_layers(item_features)

        for feat in user_input_layers.keys():
            if isinstance(user_input_layers[feat], SparseSeqInput):
                embed_input = user_input_layers[feat]
                user_input_layers[feat] = SequencePoolingLayer("mean")(embed_input)

        for feat in item_input_layers.keys():
            if isinstance(item_input_layers[feat], SparseSeqInput):
                embed_input = item_input_layers[feat]
                item_input_layers[feat] = SequencePoolingLayer("mean")(embed_input)

        user_dnn = []

    def call(self, x):
        pass
