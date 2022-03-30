from typing import List, Union, OrderedDict, Dict, Tuple
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from handyrec.layers import CustomEmbedding, SequencePoolingLayer, ValueTable
from handyrec.layers.utils import concat
from handyrec.features.utils import split_features
from handyrec.features.features import Feature, SparseFeature, SparseSeqFeature


class FeaturePool:
    def __init__(
        self,
    ) -> None:
        pass


class FeatureGroup:
    """A wrapper that consturct input, embedding, and concatenated output of a group of features

    Attributes
    ----------
    name : str
        Name of this `FeatureGroup`, should be unique.
    input_layers : OrderedDict[str, Input]
        Dictionary that stores input layers of all features.
    embd_layers : OrderedDict[str, CustomEmbedding]
        Dictionary that stores embedding layer of sparse features.
    dense_output: List[Input]
        List of input layers of dense features, to be concatenated
    sparse_output: List
        List of embeddings corredsponding to the input index
    """

    def __init__(
        self,
        name: str,
        features: List[Feature],
        l2_embd: float = 1e-6,
        pre_embd: Dict[str, Union[np.ndarray, tf.Tensor]] = None,
        pool_method: str = "mean",
    ):
        """Initialize a `FeatureGroup`

        Parameters
        ----------
        name : str
            Name of this `FeatureGroup`, should be unique.
        features : List[Feature]
            List of features in this group.
        l2_embd : float
            L2 regularization parameter for embeddings, by default `1e-6`
        pre_embd : Dict[str, Union[np.ndarray, tf.Tensor]], optional
            Dictionary of pretrained embeddings, {feature_name:embd}, by default `None`
        pool_method : str
            Pooling method for `SparseSeqFeature`s, by default `"mean"`

        Raises
        ------
        ValueError
            If the unit of a `SparseSeqFeature` is not `SparseFeature` nor `EmbdFeatureGroup`
        """
        # * check validity
        for feat in features:
            if (
                isinstance(feat, SparseSeqFeature)
                and not isinstance(feat.unit, SparseFeature)
                and not isinstance(feat.unit, EmbdFeatureGroup)
            ):
                raise ValueError(
                    """Only a `EmbdFeatureGroup` can be the unit of a `SparseSeqFeature`"""
                )

        # * initialize attributes
        self.name = name

        # * construct input and embedding layers, generate output
        self.input_layers = self.construct_inputs(features)
        self.embd_layers = self.construct_embds(features, l2_embd, pre_embd)
        # * build layers and get output
        dense, sparse, sparse_seq = split_features(features)
        self.dense_output = [self.input_layers[k] for k in dense.keys()]
        self.sparse_output = self.embedding_lookup(sparse, sparse_seq, pool_method)

    @classmethod
    def construct_inputs(cls, features: List[Feature]) -> OrderedDict[str, Input]:
        """Construct input layers pf each feature

        Parameters
        ----------
        features : List[Feature]
            List of features in this group.

        Returns
        -------
        OrderedDict[str, Input]
            Dictionary of input layers, {name: input layer}
        """
        input_layers = OrderedDict()
        dense, sparse, sparse_seq = split_features(features)

        for feat in dense.values():
            input_layers[feat.name] = Input(
                shape=(1,), name=feat.name, dtype=feat.dtype
            )

        for feat in sparse.values():
            input_layers[feat.name] = Input(
                shape=(1,), name=feat.name, dtype=feat.dtype
            )

        for feat in sparse_seq.values():
            input_layers[feat.name] = Input(
                shape=(feat.seq_len,), name=feat.name, dtype=feat.dtype
            )  # * sparse feature index seq
        return input_layers

    @classmethod
    def construct_embds(
        cls,
        features: List[Feature],
        l2_reg: float,
        pre_embd: Dict[str, Union[np.ndarray, tf.Tensor]] = None,
    ) -> OrderedDict[str, CustomEmbedding]:
        """Construct embedding layers of sparse features

        Parameters
        ----------
        features : List[Feature]
            List of features in this group.
        l2_reg : float
            L2 regularization parameter for embeddings.
        pre_embd : Dict[str, Union[np.ndarray, tf.Tensor]], optional
            Dictionary of pretrained embeddings, {feature_name:embd}, by default `None`

        Returns
        -------
        OrderedDict[str, CustomEmbedding]
            Dictionary of embedding layers, {name: embedding layer}.
        """
        embd_layers = OrderedDict()
        # * construct embedding layer of `SparseFeature`s and `SparseSeqFeature`s
        _, sparse, sparse_seq = split_features(features)

        feats_to_construct = list(sparse.values())
        for feat in sparse_seq.values():
            if not feat.is_group and feat.unit.name not in sparse.keys():
                # * append the unit of all `SparseSeqFeature`s if it is not a `EmbdFeatureGroup`
                feats_to_construct.append(feat.unit)
            elif feat.is_group:
                # * `EmbdFeatureGroup.__call__` will be called later
                embd_layers[feat.unit.name] = feat.unit

        for feat in feats_to_construct:
            weights = None
            if pre_embd and feat.name in pre_embd.keys():
                weights = pre_embd[feat.name]
            embd_layer = CustomEmbedding(
                input_dim=feat.vocab_size,
                output_dim=feat.embdding_dim,
                embeddings_regularizer=l2(l2_reg),
                trainable=feat.trainable,
                name="embd_" + feat.name,
                weights=weights,
                mask_zero=feat.name in [x.unit.name for x in sparse_seq.values()],
            )

            embd_layers[feat.name] = embd_layer
        return embd_layers

    def embedding_lookup(
        self,
        sparse_feats: List[Feature],
        sparse_seq_feats: List[Feature],
        pool_method: str = "mean",
    ) -> List[tf.Tensor]:
        """Concatenate all features in this group and output a tensor that is ready for model input.

        Parameters
        ----------
        sparse_feats : List[Feature]
            `SparseFeature` list.
        sparse_seq_feats : List[Feature]
            `SparseSeqFeature` list.
        pool_method : str, optional
            Pooling method for `SparseSeqFeature`s, by default `"mean"`.

        Returns
        -------
        List[tf.Tensor]
            Output of all features in this group, (dense output list, sparse output list)
        """

        # * Embedding output: input layer -> embedding layer (-> pooling layer)
        embd_outputs = OrderedDict()
        for name in sparse_feats.keys():
            embd_outputs[name] = self.embd_layers[name](self.input_layers[name])

        for feat in sparse_seq_feats.values():
            sparse_embd = self.embd_layers[feat.unit.name]
            seq_input = self.input_layers[feat.name]
            pool_layer = SequencePoolingLayer(pool_method)
            if feat.is_group:
                embd_seq, mask = sparse_embd(seq_input)
                embd_outputs[feat.name] = pool_layer(embd_seq, mask)
            else:
                embd_seq = sparse_embd(seq_input)
                embd_outputs[feat.name] = pool_layer(embd_seq)

        embd_output = list(embd_outputs.values())

        return embd_output


class EmbdFeatureGroup:
    """A wrapper that consturct concatenated embeddings of a group of features

    Attributes
    ----------
    name : str
        Name of this `EmbdFeatureGroup`, should be unique.
    id_name: str
        Name of the id featrue, e.g. `item_id`, `movie_id`
    id_input : Input
        An input layer of the id feature.
    embd_layers : OrderedDict[str, CustomEmbedding]
        Dictionary that stores embedding layer of sparse features.
    embedding: tf.Tensor
        Concatenated all feature values/embeddings of all ids.
    """

    def __init__(
        self,
        name: str,
        id_name: str,
        features: List[Feature],
        value_dict: Dict[str, np.ndarray],
        l2_embd: float = 1e-6,
        pre_embd: Dict[str, Union[np.ndarray, tf.Tensor]] = None,
        pool_method: str = "mean",
    ):
        """Initialize a `EmbdFeatureGroup`

        Parameters
        ----------
        name : str
            Name of this `EmbdFeatureGroup`, should be unique.
        id_name: str
            Name of the id featrue, e.g. `item_id`, `movie_id`
        features : List[Feature]
            List of features in this group.
        value_dict : Dict[str, np.ndarray]
            Dictionary contains full fature values of all ids, {feature_name: values of all ids}.
        l2_embd : float
            L2 regularization parameter for embeddings, by default `1e-6`.
        pre_embd : Dict[str, Union[np.ndarray, tf.Tensor]], optional
            Dictionary of pretrained embeddings, {feature_name:embd}, by default `None`.
        pool_method : str
            Pooling method for `SparseSeqFeature`s, by default `"mean"`.

        Raises
        ------
        ValueError
            If `id_name` is not the name of the feature in `features`.
            If the unit of a `SparseSeqFeature` is not `SparseFeature` nor `EmbdFeatureGroup`
        """
        # * check validity
        if id_name not in [x.name for x in features]:
            raise ValueError("`id_name` should be the name of a feature in `features`")
        for feat in features:
            if (
                isinstance(feat, SparseSeqFeature)
                and not isinstance(feat.unit, SparseFeature)
                and not isinstance(feat.unit, EmbdFeatureGroup)
            ):
                raise ValueError(
                    """Only a `EmbdFeatureGroup` can be the unit of a `SparseSeqFeature`"""
                )

        # * initialize attributes
        self.name = name
        self.id_name = id_name

        id_feat = {x.name: x for x in features}[id_name]
        self.id_input = Input(shape=(1,), name=id_name, dtype=id_feat.dtype)
        self.embd_layers = FeatureGroup.construct_embds(features, l2_embd, pre_embd)
        self.embedding = self.generate_embedding(features, value_dict, pool_method)

    def generate_embedding(
        self,
        features: List[Feature],
        value_dict: Dict[str, np.ndarray],
        pool_method: str = "mean",
    ) -> tf.Tensor:
        """Concatenate all features in this group and output a tensor that is ready for lookup.

        Parameters
        ----------
        features : List[Feature]
            List of features in this group.
        value_dict : Dict[str, np.ndarray]
            Dictionary contains full fature values of all ids, {feature_name: values of all ids}.
        pool_method : str, optional
            Pooling method for `SparseSeqFeature`s, by default `"mean"`.

        Returns
        -------
        tf.Tensor
            concatenated full feature value list of all ids, size: (id feature vocab_size, k)
        """
        dense, sparse, sparse_seq = split_features(features)
        # * Embedding output: input layer -> embedding layer (-> pooling layer)
        embd_outputs = OrderedDict()

        # * if this FeatureGroup is an embedding, a dense feature needs to be treated as an 1-d embedding
        for name in dense.keys():
            embd = ValueTable(value_dict[name])([])
            # ? Changing self.id_input into [] will lead to decrease of performance and converge speed, why?
            embd_outputs[name] = tf.expand_dims(embd, axis=-1)  # * (n,1)

        for name in sparse.keys():
            embd_input = ValueTable(value_dict[name])([])
            embd_outputs[name] = self.embd_layers[name](embd_input)  # * (n,d)

        for feat in sparse_seq.values():
            sparse_embd = self.embd_layers[feat.unit.name]
            seq_input = ValueTable(value_dict[feat.name])([])
            if not feat.is_group:
                embd_seq = sparse_embd(seq_input)
            else:
                embd_seq = tf.nn.embedding_lookup(sparse_embd, seq_input)
            embd_outputs[feat.name] = SequencePoolingLayer(pool_method)(embd_seq)
            embd_outputs[feat.name] = tf.squeeze(embd_outputs[feat.name])  # * (n,d)

        output = concat([], list(embd_outputs.values()))  # * shape: (n, 2d+k)

        return output

    def __call__(self, seq_input) -> Tuple[tf.Tensor]:
        output = tf.nn.embedding_lookup(self.embedding, seq_input)
        # * manually compute mask
        mask = tf.not_equal(seq_input, 0)  # (?, n)
        mask = tf.expand_dims(mask, axis=-1)  # (?, n, 1)
        tile_shape = [1] * (len(mask.shape) - 1) + [self.embedding.shape[1]]
        mask = tf.tile(mask, tile_shape)  # (?, n, output_dim)

        return output, mask

    def lookup(self) -> tf.Tensor:
        """Lookup embedding of `id_input`

        Returns
        -------
        tf.Tensor
            Embedding of `id_input`, size = (batch_size,d).
        """
        embedding = tf.nn.embedding_lookup(self.embedding, self.id_input)
        embedding = tf.squeeze(embedding, axis=1)
        return embedding


if __name__ == "__main__":
    from handyrec.features.features import DenseFeature, SparseFeature, SparseSeqFeature

    all_item_model_input = {
        "movie_id": [1, 2, 3, 4, 5, 6],
        "genres": [
            [1, 3, 0],
            [2, 4, 9],
            [11, 0, 0],
            [5, 7, 0],
            [13, 14, 15],
            [16, 17, 18],
        ],
    }

    MATCH_EMBEDDING_DIM = 64
    item_features = [
        SparseFeature("movie_id", 6, MATCH_EMBEDDING_DIM),
        SparseSeqFeature(
            SparseFeature("genre_id", 19, MATCH_EMBEDDING_DIM), "genres", 3
        ),
    ]
    item_feature_group = EmbdFeatureGroup(
        "item", "movie_id", item_features, all_item_model_input
    )

    user_features = (
        [DenseFeature("age"), DenseFeature("height")]
        + [SparseFeature("user_id", 5500, MATCH_EMBEDDING_DIM)]
        + [
            SparseSeqFeature(
                item_feature_group,
                "hist_movie_seq",
                40,
            )
        ]
    )
    user_feature_group = FeatureGroup("user", "user_id", user_features)
