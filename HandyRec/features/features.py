class Feature:
    """Base class for different types of features"""

    def __init__(self):
        pass


class DenseFeature(Feature):
    """Dense Feature class"""

    def __init__(self, name: str, dtype: str = "int32"):
        super().__init__()

        self.name = name
        self.dtype = dtype


class SparseFeature(Feature):
    """Sparse feature class"""

    def __init__(
        self, name: str, vocab_size: int, embedding_dim: int, dtype: str = "int32"
    ):
        """
        Args:
            name (str): Name of feature, each feature should have a distinct name.
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            dtype (str, optional): Data type. Defaults to 'int32'.
        """
        super().__init__()

        self.name = name
        self.vocab_size = vocab_size
        self.embdding_dim = embedding_dim
        self.dtype = dtype


class SparseSeqFeature(Feature):
    """Sparse sequence feature, e.g. item_id sequence"""

    def __init__(self, sparse_feat: SparseFeature, name: str, seq_len: int):
        """
        Args:
            sparse_feat (SparseFeature): sparse sequence feature
            name (str): feature name
            seq_len (int): sequence length
        """
        super().__init__()
        # TODO Add other params like `trainable`
        self.sparse_feat = sparse_feat
        self.name = name
        self.seq_len = seq_len
        self.dtype = "int32"
