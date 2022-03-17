from handyrec.dataset.metrics import *


def test_mapk():
    assert mapk([[1]], [[2, 1, 3, 4, 5]], 5) == 0.5
    assert mapk([[1]], [[4, 2, 3, 1, 5]], 5) == 0.25
    assert mapk([[1]], [[4, 2, 3, 1, 5, 6, 7]], 5) == 0.25


def test_recall_at_k():
    assert recall_at_k([[1]], [[2, 1, 3, 4, 5]], 5) == 1
    assert recall_at_k([[1]], [[4, 2, 3, 1, 5]], 5) == 1
    assert recall_at_k([[1]], [[4, 2, 3, 1, 5, 6, 7]], 5) == 1
