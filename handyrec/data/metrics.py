from typing import Any
import numpy as np


def _apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def map_at_k(actual: Any, predicted: Any, k: int = 12) -> float:
    """Compute mean average precision, map@k.

    Parameters
    ----------
    actual : Any
        Label.
    predicted : Any
        Predictions.
    k : int, optional
        k, by default ``12``.

    Returns
    -------
    float
        map@k.
    """
    return np.mean([_apk(a, p, k) for a, p in zip(actual, predicted)])


def _rk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = sum([1 for r in actual if r in predicted]) / len(actual)

    return score


def recall_at_k(actual: Any, predicted: Any, k: int = 12) -> float:
    """Compute recall@k.

    Parameters
    ----------
    actual : Any
        Label.
    predicted : Any
        Predictions.
    k : int, optional
        k, by default ``12``.

    Returns
    -------
    float
        recall@k.
    """
    return np.mean([_rk(a, p, k) for a, p in zip(actual, predicted)])


def hr_at_k(actual: Any, predicted: Any, k: int = 10) -> float:
    """Compute hit rate, hr@k.

    Parameters
    ----------
    actual : Any
        Label.
    predicted : Any
        Predictions.
    k : int, optional
        k, by default ``12``.

    Returns
    -------
    float
        hr@k.
    """
    count = 0
    for i, actual_i in enumerate(actual):
        for p in predicted[i][:k]:
            if p in actual_i:
                count += 1
                break
    return count / len(actual)
