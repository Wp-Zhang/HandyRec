import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Flatten
from typing import List


def _concat(inputs, axis: int = -1):
    """Concatenate list of input, handle the case when len(inputs)=1

    Args:
        inputs : list of input
        axis (int, optional): concatenate axis. Defaults to -1.
        # mask (bool, optional): whether to keep masks of input tensors. Defaults to Ture.

    Returns:
        _type_: concatenated input
    """
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


def concat(
    dense_inputs: List, embd_inputs: List, axis: int = -1, keepdims: bool = False
):
    """Concatenate dense features and embedding of sparse features together

    Args:
        dense_inputs (List): dense features
        embd_inputs (List): embedding of sparse features
        axis (int, optional): concatenate axis. Deafults to `-1`
        keepdims (bool, optional): whether to flatten all inputs before concatenating. Defaults to `False`
        # mask (bool, optional): whether to keep masks of input tensors. Defaults to Ture.
    """
    if len(dense_inputs) + len(embd_inputs) == 0:
        raise ValueError("Number of inputs should be larger than 0")

    if len(dense_inputs) > 0 and len(embd_inputs) > 0:
        dense = _concat(dense_inputs, axis)
        sparse = _concat(embd_inputs, axis)
        if not keepdims:
            dense = Flatten()(dense)
            sparse = Flatten()(sparse)

        # * Change dtype
        if dense.dtype != sparse.dtype:
            if dense.dtype.is_integer:
                dense = tf.cast(dense, sparse.dtype)
            else:
                sparse = tf.cast(sparse, dense.dtype)

        return _concat([dense, sparse], axis)

    if len(dense_inputs) > 0:
        output = _concat(dense_inputs, axis)
        if not keepdims:
            output = Flatten()(output)
        return output

    if len(embd_inputs) > 0:
        output = _concat(embd_inputs, axis)
        if not keepdims:
            output = Flatten()(output)
        return output


def sampledsoftmaxloss(y_true, y_pred):
    """Helper function for calculating sampled softmax loss

    Args:
        y_true : label
        y_pred : prediction

    Returns:
        _type_: loss
    """
    return tf.reduce_mean(y_pred)
