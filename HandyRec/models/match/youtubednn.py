import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Flatten, Lambda
from ...features import SparseFeature
from ...features.utils import split_features
from ...layers import SequencePoolingLayer, DNN, EmbeddingIndex, SampledSoftmaxLayer
from ...layers.utils import get_input_layers, get_embedding_layers
from typing import Tuple, List


class YouTubeDNN(Model):
    """YouTubeDNN model"""

    def __init__(
        self,
        user_features: List,
        item_id: SparseFeature,
        num_sampled: int = 1,
        user_dnn_hidden_units: Tuple[int] = (64, 32),
        dnn_activation: str = "relu",
        l2_dnn: float = 0,
        l2_emb: float = 1e-6,
        dnn_dropout: float = 0,
        seed: int = 2022,
    ):
        u_dense, u_sparse, u_sparse_seq = split_features(user_features)

        assert len(user_features) > 0, "Should have at least one user feature"

        # * Get input layers
        u_dense_inputs = get_input_layers(u_dense)
        u_sparse_inputs = get_input_layers(u_sparse)
        u_sparse_seq_inputs = get_input_layers(u_sparse_seq)
        item_id_input = get_input_layers([item_id])[item_id.name]

        # * Get embedding layers
        user_emb_layers = get_embedding_layers(u_sparse + u_sparse_seq, l2_reg=l2_emb)
        item_emb_layer = get_embedding_layers([item_id], l2_reg=l2_emb)[item_id.name]

        # * input layer -> embedding layer (-> pooling layer)
        for feat in u_sparse_inputs.keys() | u_sparse_seq_inputs.keys():
            user_emb_layers[feat] = user_emb_layers[feat](u_sparse_inputs[feat])
            if feat in u_sparse_seq_inputs.keys():
                seq_pool_layer = SequencePoolingLayer("mean")
                user_emb_layers[feat] = seq_pool_layer(user_emb_layers[feat])

        # * Get full item embedding
        item_index = EmbeddingIndex(list(range(item_id.vocab_size)))(item_id_input)
        item_embedding = item_emb_layer(item_index)

        # * concat input layers -> DNN
        user_dnn_input = list(u_dense_inputs.values()) + list(user_emb_layers.values())
        user_dnn_input = Flatten()(Concatenate()(user_dnn_input))
        user_dnn_output = DNN(
            hidden_units=user_dnn_hidden_units,
            activation=dnn_activation,
            output_activation="linear",
            l2_reg=l2_dnn,
            dropout_rate=dnn_dropout,
            seed=seed,
        )(user_dnn_input)

        # * sampled softmax output
        output = SampledSoftmaxLayer(num_sampled=num_sampled)(
            [item_embedding, user_dnn_output, item_id_input]
        )

        # * setup user/item input and embedding
        self.user_inputs = (
            list(u_dense_inputs.values())
            + list(u_sparse_inputs.values())
            + list(u_sparse_seq_inputs.values())
        )
        self.user_embedding = user_dnn_output

        self.item_input = item_id_input
        self.item_embedding = Lambda(
            lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1)
        )(item_id_input)

        super(YouTubeDNN, self)._init__(
            inputs=self.user_inputs + [self.item_input], outputs=output
        )

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)
