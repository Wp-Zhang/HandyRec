"""Definition of feature wrappers.
"""
from typing import List, Union, OrderedDict, Dict, Tuple
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.regularizers import l2
from handyrec.layers import CustomEmbedding, SequencePoolingLayer, ValueTable
from handyrec.layers.utils import concat
from .utils import split_features
from .type import Feature, SparseFeature, SparseSeqFeature


class FeaturePool:
    """A container that stores input layers and embedding layers for groups of features.

    Attributes
    ----------
    input_layers : OrderedDict[str, Input]
        Dictionary of inpute layers shared by groups of features.
    embd_layers : OrderedDict[str, CustomEmbedding]
        Dictionary of embedding layers shared by groups of features.
    pre_embd : Dict[str, Union[np.ndarray, tf.Tensor]], optional
        Dictionary of pretrained embeddings, {feature_name:embd}
    """

    def __init__(
        self, pre_embd: Dict[str, Union[np.ndarray, tf.Tensor]] = None
    ) -> None:
        """
        Parameters
        ----------
        pre_embd : Dict[str, Union[np.ndarray, tf.Tensor]], optional
            Pretrained embedding dictionary, by default ``None``.
        """
        self.input_layers = OrderedDict()
        self.embd_layers = OrderedDict()
        self.pool_layers = OrderedDict()

        self.pre_embd = pre_embd

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
            Input layer named as ``feat_name``.

        Raises
        ------
        AttributeError
            If layer attributes conflict with an existed input layer with the same name.
        """
        if name in self.input_layers.keys():
            layer = self.input_layers[name]

            curr_dim = layer.shape[-1]
            curr_dtype = layer.dtype.name
            new_dim = params["shape"][0]
            new_dtype = params["dtype"]
            if curr_dim != new_dim or curr_dtype != new_dtype:
                raise AttributeError(
                    f"Params of {name} conflict with an existed input layer!\n"
                    + f"\t existed shape:{curr_dim}, new shape:{new_dim}\n"
                    + f"\t existed dtype:{curr_dtype}, new dtype:{new_dtype}"
                )
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
            Embedding layer named as ``feat_name``.

        Raises
        ------
        AttributeError
            If layer attributes conflict with an existed embedding layer with the same name.
        """
        if name in self.embd_layers.keys():
            layer = self.embd_layers[name]

            old_in_dim = layer.input_dim
            old_out_dim = layer.output_dim
            new_in_dim = params["input_dim"]
            new_out_dim = params["output_dim"]
            if old_in_dim != new_in_dim or old_out_dim != new_out_dim:
                raise AttributeError(
                    f"Params of {name} conflict with an existed embedding layer!\n"
                    + f"\t existed input_dim:{old_in_dim}, new input_dim:{new_in_dim}\n"
                    + f"\t existed dtype:{old_out_dim}, new dtype:{new_out_dim}"
                )
            if params["mask_zero"] and not layer.mask_zero:
                # * always set as `True` when there is a conflict on `mask_zero`
                layer = CustomEmbedding(**params)
        else:
            layer = CustomEmbedding(**params)
        self.embd_layers[name] = layer
        return layer

    def init_pool(self, name: str, params: Dict) -> Layer:
        if name in self.pool_layers.keys():
            layer = self.pool_layers[name]
            # TODO check pool method equality
        else:
            layer = SequencePoolingLayer(**params)
            self.pool_layers[name] = layer
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
    ):
        """
        Parameters
        ----------
        name : str
            Name of this `FeatureGroup`, should be unique.
        features : List[Feature]
            List of features in this group.
        feature_pool : FeaturePool
            The `FeaturePool` instance that this `FeatureGroup` belongs to.
        l2_embd : float
            L2 regularization parameter for embeddings, by default ``1e-6``

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
        self.embd_layers = self.construct_embds(features, feature_pool, l2_embd)

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
        cls, features: List[Feature], feature_pool: FeaturePool, l2_reg: float
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
            if feature_pool.pre_embd and feat.name in feature_pool.pre_embd.keys():
                weights = feature_pool.pre_embd[feat.name]
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
            Pooling method for `SparseSeqFeature`s, by default ``"mean"``.

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

            pool_layer = self.feat_pool.init_pool(
                feat.name + "_POOL",
                {"name": feat.name + "_POOL", "method": pool_method},
            )
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
    embd_dim : int
        Output dimension of embedding, use a `Dense` layer to compress.
    id_name: str
        Name of the id featrue, e.g. ``item_id``, ``movie_id``
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
        embd_dim: int,
        l2_embd: float = 1e-6,
        pool_method: str = "mean",
    ):
        """
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
        embd_dim : int
            Output dimension of embedding, use a `Dense` layer to compress.
        l2_embd : float
            L2 regularization parameter for embeddings, by default ``1e-6``.
        pool_method : str
            Pooling method for `SparseSeqFeature`s, by default ``"mean"``.

        Raises
        ------
        ValueError
            If `id_name` is not the name of the feature in `features`.
        ValueError
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
        self.embd_dim = embd_dim

        # * init input layer of id feature for embedding lookup in the future
        id_feat = {x.name: x for x in features}[id_name]
        params = {"name": id_name, "shape": (1,), "dtype": id_feat.dtype}
        self.id_input = feature_pool.init_input(id_name, params)

        # * construct embedding layers
        self.embd_layers = FeatureGroup.construct_embds(features, feature_pool, l2_embd)

        self.features = features
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
        self._output_layer = Dense(embd_dim, name="reduce_dim")

    def get_embd(self, index: Input, compress: bool = True) -> tf.Tensor:
        """Concatenate all features in this group and output a tensor that is ready for lookup.

        Parameters
        ----------
        index : Input
            Lookup index.
        compress : bool
            Whether compress the output into a size of ``self.embd_dim``.

        Note
        ----
        The code will run successfully after replacing ``index`` with ``[]``, but there will be a
            huge decrease in performance and converge speed, WHY? **TO BE SOLVED**

        Returns
        -------
        tf.Tensor
            concatenated full feature value list of all ids, size: (id feature vocab_size, k)
        """
        # * Embedding output: input layer -> embedding layer (-> pooling layer)
        embd_outputs = OrderedDict()
        dense, sparse, sparse_seq = split_features(self.features)

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
        if compress:
            output = self._output_layer(output)  # * (n, embd_dim)
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
        output = tf.nn.embedding_lookup(embedding, index)  # * (batch, seq, embd_dim)
        if index.shape[-1] == 1:
            # * index is not sequence
            output = tf.squeeze(output, axis=1)  # * (batch, embd_dim)
        return output

    def __call__(self, seq_input: Input) -> Tuple[tf.Tensor]:
        output = self.lookup(seq_input)
        # * manually compute mask
        mask = tf.not_equal(seq_input, 0)  # (?, n)
        mask = tf.expand_dims(mask, axis=-1)  # (?, n, 1)
        tile_shape = [1] * (len(mask.shape) - 1) + [output.shape[-1]]
        mask = tf.tile(mask, tile_shape)  # (?, n, output_dim)

        return output, mask
