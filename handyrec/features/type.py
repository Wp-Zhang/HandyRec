class Feature:
    """Base class for different types of features

    Attributes
    ----------
    name : str
        Name of feature, each feature should have a distinct name.
    dtype : str
        Data type.
    """

    def __init__(self, name: str, dtype: str):
        """Initialize a `Feature`.

        Parameters
        ----------
        name : str
            Name of feature, each feature should have a distinct name.
        dtype : str
            Data type.
        """
        self.name = name
        self.dtype = dtype


class DenseFeature(Feature):
    """Feature class for dense features.

    Attributes
    ----------
    name : str
        Name of feature, each feature should have a distinct name.
    dtype : str, optional
        Data type, by default ``"int32"``.
    """

    def __init__(self, name: str, dtype: str = "int32"):
        """Initialize a `DenseFeature`.

        Parameters
        ----------
        name : str
            Name of feature, each feature should have a distinct name.
        dtype : str, optional
            Data type, by default ``"int32"``.
        """
        super().__init__(name, dtype)


class SparseFeature(Feature):
    """Feature class for sparse features.

    Attributes
    ----------
    name : str
        Name of feature, each feature should have a distinct name.
    vocab_size : int
        Vocabulary size.
    embedding_dim : int
        Embedding dimension.
    trainable : bool, optional
        Whether embedding is trainable or not, by default ``True``.
    dtype : str, optional
        Data type, by default ``"int32"``.
    """

    def __init__(
        self,
        name: str,
        vocab_size: int,
        embedding_dim: int,
        trainable: bool = True,
        dtype: str = "int32",
    ):
        """Initialize a `SparseFeature`.

        Parameters
        ----------
        name : str
            Name of feature, each feature should have a distinct name.
        vocab_size : int
            Vocabulary size.
        embedding_dim : int
            Embedding dimension.
        trainable : bool, optional
            Whether embedding is trainable or not, by default ``True``.
        dtype : str, optional
            Data type, by default ``"int32"``.
        """
        super().__init__(name, dtype)
        self.vocab_size = vocab_size
        self.embdding_dim = embedding_dim
        self.trainable = trainable


class SparseSeqFeature(Feature):
    """Feature class for sparse sequence features, e.g. watch history of movies.

    Attributes
    ----------
    unit :
        Unit of sequence, can be a instance of `SparseFeature` or `EmbdFeatureGroup`.
    name : str
        Name of feature, each feature should have a distinct name.
    seq_len : int
        Sequence length.
    dtype : str
        Set as ``"int32"``.
    is_group : bool
        Whether `unit` is a instance of `EmbdFeatureGroup`.
    """

    def __init__(self, unit, name: str, seq_len: int):
        """Initialize a `SparseSeqFeature`.

        Parameters
        ----------
        unit :
            Unit of sequence, can be a instance of `SparseFeature` or `EmbdFeatureGroup`.
        name : str
            Name of feature, each feature should have a distinct name.
        seq_len : int
            Sequence length.
        """
        super().__init__(name, "int32")
        self.unit = unit
        self.seq_len = seq_len
        # * unit is a feature group if it's not a SparseFeature
        self.is_group = not isinstance(unit, SparseFeature)
