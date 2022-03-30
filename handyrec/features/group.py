from typing import List, Union, OrderedDict, Dict, Tuple
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from handyrec.layers import CustomEmbedding, SequencePoolingLayer, ValueTable
from handyrec.layers.utils import concat
from handyrec.features.utils import split_features
from handyrec.features.type import Feature, SparseFeature, SparseSeqFeature


class FeaturePool:
    """A container to store input layers and embedding layers for groups of features.

    Attributes
    ----------
    input_layers : OrderedDict[str, Input]
        Dictionary of inpute layers shared by groups of features.
    embd_layers : OrderedDict[str, CustomEmbedding]
        Dictionary of embedding layers shared by groups of features.
    """

    def __init__(self) -> None:
        self.input_layers = OrderedDict()
        self.embd_layers = OrderedDict()

    def check_input(self, feat_name: str) -> bool:
        """Check if the input layer of a feature is already created.

        Parameters
        ----------
        feat_name : str
            Name of feature.

        Returns
        -------
        bool
            Whether the input layer of the feature is created or not.
        """
        return feat_name in self.input_layers.keys()

    def check_embd(self, feat_name: str) -> bool:
        """Check if the embedding layer of a feature is already created.

        Parameters
        ----------
        feat_name : str
            Name of feature.

        Returns
        -------
        bool
            Whether the embedding layer of the feature is created or not.
        """
        return feat_name in self.embd_layers.keys()

    def init_input(self, name: str, params: Dict) -> Input:
        """Get an existed or construct a new shared input layer by name.

        Parameters
        ----------
        name : str
            Name of layer.
        params : Dict
            Input layer parameters.

        Returns
        -------
        Input
            Input layer named as `feat_name`.

        Raises
        ------
        AttributeError
            If layer attributes conflict with an existed input layer with the same name.
        """
        if name in self.input_layers.keys():
            layer = self.input_layers[name]
            if (
                layer.shape[-1] != params["shape"] or layer.dtype != params["dtype"]
            ):  # TODO real check
                raise AttributeError("Params conflict with an existed input layer!")
        else:
            layer = Input(**params)
            self.input_layers[name] = layer
        return layer

    def init_embd(self, name: str, params: Dict) -> CustomEmbedding:
        """Get an existed or construct a new shared embedding layer by name.

        Parameters
        ----------
        name : str
            Name of layer.
        params : Dict
            Input layer parameters.

        Returns
        -------
        CustomEmbedding
            Embedding layer named as `feat_name`.

        Raises
        ------
        AttributeError
            If layer attributes conflict with an existed embedding layer with the same name.
        """
        if name in self.embd_layers.keys():
            layer = self.embd_layers[name]
            if params["mask_zero"]:
                # * always set as `True` when there is a conflict on `mask_zero`
                layer.mask_zero = True
            if (
                layer.input_dim != params["input_dim"]
                or layer.output_dim != params["output_dim"]
            ):
                raise AttributeError("Params conflict with an existed embedding layer!")
        else:
            layer = CustomEmbedding(**params)
            self.embd_layers[name] = layer
        return layer

    def add_input(self, input_layer: Input) -> None:
        """Add an shared input layer.

        Parameters
        ----------
        input_layer : Input
            An input layer.
        """
        if input_layer.name in self.input_layers.keys():
            pass
        else:
            self.input_layers[input_layer.name] = input_layer

    def add_embd(self, embd_layer: CustomEmbedding) -> None:
        """Add an shared embedding layer.

        Parameters
        ----------
        embd_layer : CustomEmbedding
            An embedding layer.
        """
        self.embd_layers[embd_layer.name] = embd_layer


class FeatureGroup:
    """A wrapper that consturct input, embedding, and concatenated output of a group of features

    Attributes
    ----------
    name : str
        Name of this `FeatureGroup`, should be unique.
    features : List[Feature]
        List of features in this group.
    feature_pool : FeaturePool
        The `FeaturePool` instance that this `FeatureGroup` belongs to.
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
        feature_pool: FeaturePool,
        l2_embd: float = 1e-6,
        pre_embd: Dict[str, Union[np.ndarray, tf.Tensor]] = None,
    ):
        """Initialize a `FeatureGroup`

        Parameters
        ----------
        name : str
            Name of this `FeatureGroup`, should be unique.
        features : List[Feature]
            List of features in this group.
        feature_pool : FeaturePool
            The `FeaturePool` instance that this `FeatureGroup` belongs to.
        l2_embd : float
            L2 regularization parameter for embeddings, by default `1e-6`
        pre_embd : Dict[str, Union[np.ndarray, tf.Tensor]], optional
            Dictionary of pretrained embeddings, {feature_name:embd}, by default `None`

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
        self.features = features
        self.feat_pool = feature_pool

        # * construct input and embedding layers, generate output
        self.input_layers = self.construct_inputs(features, feature_pool)
        self.embd_layers = self.construct_embds(
            features, feature_pool, l2_embd, pre_embd
        )

    @classmethod
    def construct_inputs(
        cls, features: List[Feature], feature_pool: FeaturePool
    ) -> OrderedDict[str, Input]:
        """Construct input layers pf each feature

        Parameters
        ----------
        features : List[Feature]
            List of features in this group.
        feature_pool : FeaturePool
            The `FeaturePool` instance that this `FeatureGroup` belongs to.

        Returns
        -------
        OrderedDict[str, Input]
            Dictionary of input layers, {name: input layer}
        """
        input_layers = OrderedDict()

        for feat in features:
            dim = feat.seq_len if isinstance(feat, SparseSeqFeature) else 1
            params = {"name": feat.name, "shape": (dim,), "dtype": feat.dtype}
            input_layer = feature_pool.init_input(feat.name, params)
            input_layers[feat.name] = input_layer

        return input_layers

    @classmethod
    def construct_embds(
        cls,
        features: List[Feature],
        feature_pool: FeaturePool,
        l2_reg: float,
        pre_embd: Dict[str, Union[np.ndarray, tf.Tensor]] = None,
    ) -> OrderedDict[str, CustomEmbedding]:
        """Construct embedding layers of sparse features

        Parameters
        ----------
        features : List[Feature]
            List of features in this group.
        feature_pool : FeaturePool
            The `FeaturePool` instance that this `FeatureGroup` belongs to.
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
        _, sparse, sparse_seq = split_features(features)
        # * unit of a `SparseSeqFeature` should be a embedding layer with `mask_zero` setting as `True`
        mask_embds = [x.unit.name for x in sparse_seq.values()]

        feats_to_construct = list(sparse.values())
        for feat in sparse_seq.values():
            if not feat.is_group and feat.unit.name not in sparse.keys():
                # * append the unit of a `SparseSeqFeature` if it is not a `EmbdFeatureGroup`
                feats_to_construct.append(feat.unit)
            elif feat.is_group:
                # * `EmbdFeatureGroup.__call__` will be called later
                embd_layers[feat.unit.name] = feat.unit
                feature_pool.add_embd(feat.unit)

        for feat in feats_to_construct:
            weights = None
            if pre_embd and feat.name in pre_embd.keys():
                weights = pre_embd[feat.name]
            params = {
                "name": "embd_" + feat.name,
                "input_dim": feat.vocab_size,
                "output_dim": feat.embdding_dim,
                "embeddings_regularizer": l2(l2_reg),
                "trainable": feat.trainable,
                "weights": weights,
                "mask_zero": feat.name in mask_embds,
            }

            embd_layer = feature_pool.init_embd(feat.name, params)
            embd_layers[feat.name] = embd_layer
        return embd_layers

    def embedding_lookup(
        self,
        pool_method: str = "mean",
    ) -> Tuple[List[Input], List[tf.Tensor]]:
        """Concatenate all features in this group and output a tensor that is ready for model input.

        Parameters
        ----------
        pool_method : str, optional
            Pooling method for `SparseSeqFeature`s, by default `"mean"`.

        Returns
        -------
        Tuple[List[Input], List[tf.Tensor]]
            Output of all features in this group, (dense output list, sparse output list)
        """
        dense, sparse, sparse_seq = split_features(self.features)
        dense_output = [self.input_layers[k] for k in dense.keys()]

        # * Embedding output: input layer -> embedding layer (-> pooling layer)
        embd_outputs = OrderedDict()
        for name in sparse.keys():
            embd_outputs[name] = self.embd_layers[name](self.input_layers[name])

        for feat in sparse_seq.values():
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

        return dense_output, embd_output


class EmbdFeatureGroup:
    """A wrapper that consturct concatenated embeddings of a group of features

    Attributes
    ----------
    name : str
        Name of this `EmbdFeatureGroup`, should be unique.
    feature_pool : FeaturePool
        The `FeaturePool` instance that this `FeatureGroup` belongs to.
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
        feature_pool: FeaturePool,
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
        feature_pool : FeaturePool
            The `FeaturePool` instance that this `FeatureGroup` belongs to.
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
        self.feat_pool = feature_pool

        # * init input layer of id feature for embedding lookup in the future
        id_feat = {x.name: x for x in features}[id_name]
        params = {"name": id_name, "shape": (1,), "dtype": id_feat.dtype}
        self.id_input = feature_pool.init_input(id_name, params)

        # * construct embedding layers
        self.embd_layers = FeatureGroup.construct_embds(
            features, feature_pool, l2_embd, pre_embd
        )

        self._features = features
        self._value_dict = value_dict
        self._layers = {}
        for feat in features:
            self._layers[feat.name] = ValueTable(
                value_dict[feat.name], name=feat.name + "_list"
            )
            if isinstance(feat, SparseSeqFeature):
                self._layers[feat.name + "_pool"] = SequencePoolingLayer(
                    pool_method, name=feat.name + "_" + pool_method
                )

    def get_embd(self, index: Input) -> tf.Tensor:
        """Concatenate all features in this group and output a tensor that is ready for lookup.

        Parameters
        ----------
        index : Input
            Lookup index.
        pool_method : str, optional
            Pooling method for `SparseSeqFeature`s, by default `"mean"`.

        Note
        ----
            The code will run successfully after replacing `index` with `[]`, but there will be a
                huge decrease in performance and converge speed, WHY? **TO BE SOLVED**

        Returns
        -------
        tf.Tensor
            concatenated full feature value list of all ids, size: (id feature vocab_size, k)
        """
        # * Embedding output: input layer -> embedding layer (-> pooling layer)
        embd_outputs = OrderedDict()
        dense, sparse, sparse_seq = split_features(self._features)

        # * a dense feature needs to be treated as an 1-d embedding
        for name in dense.keys():
            embd = self._layers[name](index)
            embd_outputs[name] = tf.expand_dims(embd, axis=-1)  # * (n,1)

        for name in sparse.keys():
            embd_input = self._layers[name](index)
            embd_outputs[name] = self.embd_layers[name](embd_input)  # * (n,d)

        for name, feat in sparse_seq.items():
            embd_input = self._layers[name](index)
            embd_layer = self.embd_layers[feat.unit.name]
            if not feat.is_group:
                embd_seq = embd_layer(embd_input)
                embd_outputs[name] = self._layers[name + "_pool"](embd_seq)
            else:
                # * __call__ method of `EmbdFeatureGrouop` will also return manually computed mask
                embd_seq, mask = embd_layer(embd_input)
                embd_outputs[name] = self._layers[name + "_pool"](embd_seq, mask)
            embd_outputs[name] = tf.squeeze(embd_outputs[name])  # * (n,d)

        output = concat([], list(embd_outputs.values()))  # * (n, 2d+k)

        return output

    def lookup(self, index: Input) -> tf.Tensor:
        """Lookup all feature values by given id.

        Parameters
        ----------
        index : Input
            Target ids.

        Returns
        -------
        tf.Tensor
            Concatenated feature values of given ids.
        """
        embedding = self.get_embd(index)
        output = tf.nn.embedding_lookup(embedding, index)  # * (batch, seq, 2d+k)
        if index.shape[-1] == 1:
            # * index is not sequence
            output = tf.squeeze(output, axis=1)  # * (batch, 2d+k)
        return output

    def __call__(self, seq_input: Input) -> Tuple[tf.Tensor]:
        output = self.lookup(seq_input)
        # * manually compute mask
        mask = tf.not_equal(seq_input, 0)  # (?, n)
        mask = tf.expand_dims(mask, axis=-1)  # (?, n, 1)
        tile_shape = [1] * (len(mask.shape) - 1) + [output.shape[-1]]
        mask = tf.tile(mask, tile_shape)  # (?, n, output_dim)

        return output, mask


if __name__ == "__main__":
    from handyrec.features.type import DenseFeature, SparseFeature, SparseSeqFeature

    feat_pool = FeaturePool()
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
        "item", "movie_id", item_features, feat_pool, all_item_model_input
    )

    user_features = (
        [DenseFeature("age"), DenseFeature("height")]
        + [SparseFeature("user_id", 5500, MATCH_EMBEDDING_DIM)]
        + [
            SparseSeqFeature(
                SparseFeature("movie_id", 6, MATCH_EMBEDDING_DIM),
                "hist_movie_seq",
                40,
            )
        ]
    )
    user_feature_group = FeatureGroup("user", user_features, feat_pool)
    dense_output, sparse_output = user_feature_group.embedding_lookup()
    print(user_feature_group.input_layers)
    print(user_feature_group.embd_layers)

    print(item_feature_group.embd_layers)
