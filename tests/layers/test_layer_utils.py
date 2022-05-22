from handyrec.layers.utils import concat, sampledsoftmaxloss, get_activation_layer
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding
import pytest


def test_concat_func():
    dense_inputs = [
        Input(shape=(1,), dtype="int32"),
        Input(shape=(1,), dtype="float32"),
    ]
    embd_inputs = [Embedding(10, 64)(Input(shape=(1,))) for _ in range(3)]

    # * case 1
    with pytest.raises(ValueError) as e_info:
        concat([], [])
    assert e_info.type is ValueError

    # * case 2
    output = concat(dense_inputs, embd_inputs, keepdims=False)
    assert output.shape[0] is None and output.shape[1] == 2 * 1 + 64 * 3

    # * case 3
    # dense_inputs2 = [Input(shape=(1, 1)), Input(shape=(1, 1))]
    output = concat([], embd_inputs, axis=1, keepdims=True)
    assert output.shape[0] is None and output.shape[1:] == (3, 64)

    # * case 4
    output = concat([dense_inputs[0]], [])
    assert output.shape[0] is None and output.shape[1] == 1

    # * case 5
    output = concat([], embd_inputs)
    assert output.shape[0] is None and output.shape[1] == 64 * 3


def test_sampledsoftmaxloss_func():
    y_pred = [1, 2, 3]
    assert sampledsoftmaxloss([], y_pred) == 2


def test_get_activation_layer_func():
    get_activation_layer("dice")
    get_activation_layer("sigmoid")
