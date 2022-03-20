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


def mapk(actual, predicted, k=12):
    return np.mean([_apk(a, p, k) for a, p in zip(actual, predicted)])


def _rk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = sum([1 for r in actual if r in predicted]) / len(actual)

    return score


def recall_at_k(actual, predicted, k=12):
    return np.mean([_rk(a, p, k) for a, p in zip(actual, predicted)])


def hr_at_k(label, pred, k=10):
    count = 0
    for i in range(len(label)):
        for p in pred[i][:k]:
            if p in label[i]:
                count += 1
                break
    return count / len(label)
