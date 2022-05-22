from handyrec.data.metrics import *


def test_mapk():
    assert map_at_k([[1]], [[2, 1, 3, 4, 5]], 5) == 0.5
    assert map_at_k([[1]], [[4, 2, 3, 1, 5]], 5) == 0.25
    assert map_at_k([[1]], [[4, 2, 3, 1, 5, 6, 7]], 5) == 0.25


def test_recall_at_k():
    assert recall_at_k([[1]], [[2, 1, 3, 4, 5]], 5) == 1
    assert recall_at_k([[1]], [[4, 2, 3, 1, 5]], 5) == 1
    assert recall_at_k([[1, 8]], [[4, 2, 3, 1, 5, 6, 7]], 5) == 0.5


def test_hr_at_k():
    assert hr_at_k([[1]], [[2, 1, 3, 4, 5]], 5) == 1
    assert hr_at_k([[1], [3]], [[4, 2, 3, 1, 5, 6, 7], [2, 4, 6, 8, 10, 3]], 5) == 0.5
