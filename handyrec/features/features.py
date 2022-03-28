class Feature:
    """Base class for different types of features

    Args:
        name (str): Name of feature, each feature should have a distinct name.
        dtype (str, optional): Data type.
        group (str, optional): Group name.
    """

    def __init__(self, name: str, dtype: str, group: str):
        self.name = name
        self.dtype = dtype
        self.group = group


class DenseFeature(Feature):
    """Dense Feature class

    Args:
        name (str): Name of feature, each feature should have a distinct name.
        dtype (str, optional): Data type. Defaults to 'int32'.
        group (str, optional): Group name. Defaults to 'default'.
    """

    def __init__(self, name: str, dtype: str = "int32", group: str = "default"):
        super().__init__(name, dtype, group)


class SparseFeature(Feature):
    """Sparse feature class

    Args:
        name (str): Name of feature, each feature should have a distinct name.
        vocab_size (int): Vocabulary size.
        embedding_dim (int): Embedding dimension.
        trainable (bool, optional): Whether embedding is trainable or not. Defaults to True
        dtype (str, optional): Data type. Defaults to 'int32'.
        group (str, optional): Group name. Defaults to 'default'.
    """

    def __init__(
        self,
        name: str,
        vocab_size: int,
        embedding_dim: int,
        trainable: bool = True,
        dtype: str = "int32",
        group: str = "default",
    ):
        super().__init__(name, dtype, group)
        self.vocab_size = vocab_size
        self.embdding_dim = embedding_dim
        self.trainable = trainable


class SparseSeqFeature(Feature):
    """Sparse sequence feature, e.g. item_id sequence

    Args:
        sparse_feat (SparseFeature): sequence unit
        name (str): feature name
        seq_len (int): sequence length
        group (str, optional): Group name. Defaults to 'default'.
    """

    def __init__(
        self,
        sparse_feat: SparseFeature,
        name: str,
        seq_len: int,
        group: str = "default",
    ):
        super().__init__(name, "int32", group)
        self.sparse_feat = sparse_feat
        self.seq_len = seq_len
        self.group = group
