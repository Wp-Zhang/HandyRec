"""Contains layer-related utility functions.
"""
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Flatten, Layer, Activation
from typing import List, Any
from .activation import Dice


def _concat(inputs: List, axis: int = -1) -> tf.Tensor:
    """Concatenate list of input, handle the case when `len(inputs) = 1`.

    Parameters
    ----------
    inputs : List
        List of input
    axis : int, optional
        Concatenate axis, by default ``-1``.

    Returns
    -------
    tf.Tensor
        Concatenated input.
    """
    if len(inputs) == 1:
        return inputs[0]
    else:
        # * change dtypes
        has_integer = False
        has_other = False
        for tensor in inputs:
            if tensor.dtype.is_integer:
                has_integer = True
            else:
                has_other = True
        if has_other and has_integer:
            inputs = [tf.cast(tensor, tf.float32) for tensor in inputs]
        return Concatenate(axis=axis)(inputs)


def concat(
    dense_inputs: List, embd_inputs: List, axis: int = -1, keepdims: bool = False
) -> tf.Tensor:
    """Concatenate dense features and embedding of sparse features together.

    Parameters
    ----------
    dense_inputs : List
        Dense features.
    embd_inputs : List
        Embedding of sparse features.
    axis : int, optional
        Concatenate axis, by default ``-1``.
    keepdims : bool, optional
        Whether flatten all inputs before concatenating or not, by default ``False``.

    Returns
    -------
    tf.Tensor
        Concatenated input.

    Raises
    ------
    ValueError
        If no tensor is provided.
    """

    if len(dense_inputs) + len(embd_inputs) == 0:
        raise ValueError("Number of inputs should be larger than 0")

    if len(dense_inputs) > 0 and len(embd_inputs) > 0:
        dense = _concat(dense_inputs, axis)
        sparse = _concat(embd_inputs, axis)
        if not keepdims:
            dense = Flatten()(dense)
            sparse = Flatten()(sparse)

        # # * Change dtype
        # if dense.dtype != sparse.dtype:
        #     if dense.dtype.is_integer:
        #         dense = tf.cast(dense, sparse.dtype)
        #     else:
        #         sparse = tf.cast(sparse, dense.dtype)

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


def sampledsoftmaxloss(y_true, y_pred) -> Any:
    """Helper function for calculating sampled softmax loss.

    Parameters
    ----------
    y_true
        Label.
    y_pred
        Prediction.

    Returns
    -------
    Any
        Sampled softmax loss.
    """
    return tf.reduce_mean(y_pred)


def get_activation_layer(activation: str) -> Layer:
    """Return a activation layer by input activation function.

    Parameters
    ----------
    activation : str
        Activation function name.

    Returns
    -------
    Layer
        An activation layer.
    """
    if activation == "dice":
        return Dice()
    else:
        return Activation(activation)
