from tensorflow.keras import Model

# from tensorflow.keras.layers import Dense
# from tensorflow.keras import Sequential
# from ...features import SparseFeature, DenseFeature, SparseSeqFeature
# from ...features.utils import split_features
# from ...layers import SparseSeqInput, SequencePoolingLayer
# from ...layers.utils import construct_input_layers


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

    def call(self, x):
        pass

    def get_config(self):
        return super(DSSM, self).get_config()
