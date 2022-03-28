import faiss
import numpy as np

# from typing import List, Dict, Any

# from handyrec.features.utils import split_features
# from handyrec.layers import SequencePoolingLayer
# from handyrec.layers.utils import construct_input_layers, construct_embedding_layers

# from collections import OrderedDict


def search_embedding(
    embd_dim: int,
    item_embd: np.ndarray,
    user_embd: np.ndarray,
    item_list: np.ndarray,
    n: int,
    gpu: bool = False,
) -> np.ndarray:
    """Search top n similar item embeddings for each user embedding

    Args:
        embd_dim (int): embedding dimension
        item_embd (np.ndarray): item embedding
        user_embd (np.ndarray): user embedding
        user_ids (Iterable): list of target users
        item_list (np.ndarray): full item numpy array, has same length with `item_embd`
        n (int): number of candidate items for each user
        gpu (bool, optional): use gpu to search. Defaults to False.

    Returns:
        np.array: search result. (NUM_USERS x n)
    """
    index = faiss.IndexFlatIP(embd_dim)
    index.add(item_embd)

    _, result = index.search(np.ascontiguousarray(user_embd), n)
    candidates = []
    for i in range(user_embd.shape[0]):
        pred = item_list[result[i]].tolist()
        candidates.append(pred)
    candidates = np.array(candidates)

    return candidates


# class FeatureLookupTable:
#     """Store full item/user feature values.
#     Input: item/user id (batch_size,1)
#     Output: item/user feature list [(batch_size,1,1),(batch_size,1,k),(batch_size,m,k)]
#     """

#     def __init__(
#         self,
#         name: str,
#         features: List[Any],
#         feature_dict: Dict,
#         l2_emb: float = 1e-6,
#         pool_method: str = "mean",
#     ):
#         self.name = name
#         self.features = features
#         self.feature_dict = feature_dict
#         self.l2_emb = l2_emb
#         self.pool_method = pool_method
#         self._build()

#     def _build(self):
#         # * Group features by their types
#         dense, sparse, sparse_seq = split_features(self.features)

#         # * Get input and embedding layers
#         input_layers = construct_input_layers(self.features)
#         embd_layers = construct_embedding_layers(self.features, self.l2_emb)

#         # * Embedding output: input layer -> embedding layer (-> pooling layer)
#         embd_outputs = OrderedDict()
#         for feat in sparse.keys():
#             embd_outputs[feat] = embd_layers[feat](input_layers[feat])
#         for feat in sparse_seq.values():
#             sparse_emb = embd_layers[feat.sparse_feat.name]
#             seq_input = input_layers[feat.name]
#             embd_outputs[feat.name] = SequencePoolingLayer(self.pool_method)(
#                 sparse_emb(seq_input)
#             )

#     def call(self, inputs):
#         pass
