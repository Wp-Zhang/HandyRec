class Feature:
    """Base class for different types of features

    Args:
        name (str): Name of feature, each feature should have a distinct name.
        dtype (str, optional): Data type.
    """

    def __init__(self, name: str, dtype: str):
        self.name = name
        self.dtype = dtype


class DenseFeature(Feature):
    """Dense Feature class

    Args:
        name (str): Name of feature, each feature should have a distinct name.
        dtype (str, optional): Data type. Defaults to 'int32'.
    """

    def __init__(self, name: str, dtype: str = "int32"):
        super().__init__(name, dtype)


class SparseFeature(Feature):
    """Sparse feature class

    Args:
        name (str): Name of feature, each feature should have a distinct name.
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Embedding dimension.
        trainable (bool, optional): Whether embedding is trainable or not. Defaults to True
        dtype (str, optional): Data type. Defaults to 'int32'.
    """

    def __init__(
        self,
        name: str,
        vocab_size: int,
        embedding_dim: int,
        trainable: bool = True,
        dtype: str = "int32",
    ):
        super().__init__(name, dtype)
        self.vocab_size = vocab_size
        self.embdding_dim = embedding_dim
        self.trainable = trainable


class SparseSeqFeature(Feature):
    """Sparse sequence feature, e.g. item_id sequence

    Args:
        unit (Union[SparseFeature, FeatureGroup]): sequence unit
        name (str): feature name
        seq_len (int): sequence length
    """

    def __init__(self, unit, name: str, seq_len: int):
        super().__init__(name, "int32")
        self.unit = unit
        self.seq_len = seq_len
        # * unit is a feature group if it's not a SparseFeature
        self.is_group = not isinstance(unit, SparseFeature)
