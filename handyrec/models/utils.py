from xmlrpc.client import boolean
import faiss
import numpy as np


def search_embedding(
    embd_dim: int,
    item_embd: np.ndarray,
    user_embd: np.ndarray,
    item_list: np.ndarray,
    n: int,
    gpu: boolean = False,
) -> np.ndarray:
    """Search top n similar item embeddings for each user embedding

    Args:
        embd_dim (int): embedding dimension
        item_embd (np.ndarray): item embedding
        user_embd (np.ndarray): user embedding
        user_ids (Iterable): list of target users
        item_list (np.ndarray): full item numpy array, has same length with `item_embd`
        n (int): number of candidate items for each user
        gpu (boolean, optional): use gpu to search. Defaults to False.

    Returns:
        np.array: search result. (NUM_USERS x n)
    """
    index = faiss.IndexFlatIP(embd_dim)
    index.add(item_embd)

    D, I = index.search(np.ascontiguousarray(user_embd), n)
    candidates = []
    for i in range(user_embd.shape[0]):
        pred = item_list[I[i]].tolist()
        candidates.append(pred)
    candidates = np.array(candidates)

    return candidates
