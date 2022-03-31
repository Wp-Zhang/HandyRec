"""Contains model-related utility functions.
"""
import faiss
import numpy as np


def search_embedding(
    embd_dim: int,
    item_embd: np.ndarray,
    user_embd: np.ndarray,
    item_list: np.ndarray,
    n: int,
    gpu: bool = False,
) -> np.ndarray:
    """Use faiss to earch top n similar item embeddings for each user embedding.

    Parameters
    ----------
    embd_dim : int
        Embedding dimension.
    item_embd : np.ndarray
        Item embedding.
    user_embd : np.ndarray
        User embeding.
    item_list : np.ndarray
        Full item numpy array, has the same length as `item_embd`.
    n : int
        Number of candidate items for each user.
    gpu : bool, optional
        Whether use gpu to search, by default `False`.

    Returns
    -------
    np.ndarray
        Search result, shape: (NUM_USERS, n)
    """
    index = faiss.IndexFlatIP(embd_dim)
    index.add(item_embd)

    if gpu:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    _, result = index.search(np.ascontiguousarray(user_embd), n)
    candidates = []
    for i in range(user_embd.shape[0]):
        pred = item_list[result[i]].tolist()
        candidates.append(pred)
    candidates = np.array(candidates)

    return candidates
