from typing import List, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def gen_sequence(
    df: pd.DataFrame,
    uid_name: str,
    unit_name: str,
    seq_len: int,
    padding: str = "pre",
    truncating: str = "pre",
    value: Any = 0,
) -> List[List[Any]]:
    """Generate sequence feature baseed on given unit feature and sequence length.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing unit feature.
    uid_name : str
        Name of the column containing user id.
    unit_name: str
        The name of the unit feature.
    seq_len: int
        The maximum length of the sequence.
    padding: str
        The padding method, should be one of {``post``, ``pre``}.
    truncating: str
        The truncating method, should be one of {``post``, ``pre``}.
    valud: Any
        The value to be padded.

    Returns
    -------
    List[List[Any]]
        Generated sequence.
    """

    seq = np.zeros((df.shape[0], seq_len), dtype=df[unit_name].dtype)
    p = 0
    for _, hist in tqdm(df.groupby(uid_name), f"Generate {unit_name} sequence"):
        hist = hist[unit_name].tolist()
        hists = [hist[max(0, i - seq_len) : i] for i in range(len(hist))]
        seq[p : p + len(hists), :] = pad_sequences(
            hists, maxlen=seq_len, padding=padding, truncating=truncating, value=value
        )
        p += len(hists)

    return seq.tolist()  # ! may lead to memory error, need to be fixed
