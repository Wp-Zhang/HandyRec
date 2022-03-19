from typing import OrderedDict, Tuple, List, Any

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
import warnings

from ...features.utils import split_features
from ...layers import SequencePoolingLayer, DNN, FM
from ...layers.utils import (
    construct_input_layers,
    construct_embedding_layers,
    concat_inputs,
)


def DeepFM(
    fm_features: List[Any],
    dnn_features: List[Any],
    dnn_hidden_units: Tuple[int] = (64, 32, 1),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    l2_dnn: float = 0,
    l2_emb: float = 1e-6,
    task: str = "binary",
    seed: int = 2022,
):
    """Implementation of DeepFM

    Args:
        fm_features (List[Any]): input feature list for FM
        dnn_features (List[Any]): input feature list for DNN
        dnn_hidden_units (Tuple[int], optional): DNN structure. Defaults to (64, 32).
        dnn_activation (str, optional): DNN activation function. Defaults to "relu".
        dnn_dropout (float, optional): DNN dropout ratio. Defaults to 0..
        l2_dnn (float, optional): DNN l2 regularization param. Defaults to 0.
        l2_emb (float, optional): embedding l2 regularization param. Defaults to 1e-6.
        task (str, optional): model task, should be `binary` or `regression`. Defaults to `binary`
        seed (int, optional): random seed of dropout. Defaults to 2022.
    """
    if len(fm_features) < 1:
        raise ValueError("Should have at least one feature for FM")
    if len(dnn_features) < 1:
        raise ValueError("Should have at least one feature for DNN")
    if dnn_hidden_units[-1] != 1:
        raise ValueError("Output size of dnn should be 1")

    # * Group features by their types
    fm_dense_f, fm_sparse_f, fm_sparse_seq_f = split_features(fm_features)
    dnn_dense_f, dnn_sparse_f, dnn_sparse_seq_f = split_features(dnn_features)
    _, sparse_f, sparse_seq_f = split_features(fm_features + dnn_features)

    # * Get input and embedding layers
    input_layers = construct_input_layers(fm_features + dnn_features)
    embd_layers = construct_embedding_layers(fm_features + dnn_features, l2_emb)

    # * Embedding output: input layer -> embedding layer (-> pooling layer)
    embd_outputs = OrderedDict()
    for feat in sparse_f.keys():
        embd_outputs[feat] = embd_layers[feat](input_layers[feat])
    for feat in sparse_seq_f.values():
        sparse_emb = embd_layers[feat.sparse_feat.name]
        seq_input = input_layers[feat.name]
        embd_outputs[feat.name] = SequencePoolingLayer("mean")(sparse_emb(seq_input))

    # * Concat input layers -> DNN, FM
    dnn_input = concat_inputs(
        [input_layers[k] for k in dnn_dense_f.keys()],
        [embd_outputs[k] for k in list(dnn_sparse_f.keys())]
        + [embd_outputs[k] for k in list(dnn_sparse_seq_f.keys())],
    )

    if len(fm_dense_f) > 0:
        warnings.warn(
            "FM part doesn't support dense featrue now, dense features will be ignored"
        )

    fm_input = concat_inputs(
        [],  # [input_layers[k] for k in fm_dense_f.keys()],
        [embd_outputs[k] for k in list(fm_sparse_f.keys())]
        + [embd_outputs[k] for k in list(fm_sparse_seq_f.keys())],
        axis=1,
        keepdims=True,
        mask=False,
    )

    dnn_output = DNN(
        hidden_units=dnn_hidden_units,
        activation=dnn_activation,
        output_activation="linear",
        l2_reg=l2_dnn,
        dropout_rate=dnn_dropout,
        seed=seed,
    )(dnn_input)
    fm_output = FM()(fm_input)

    # * Output
    output = dnn_output + fm_output
    if task == "binary":
        output = Activation("sigmoid")(output)

    # * Construct model
    model = Model(inputs=list(input_layers.values()), outputs=output)

    return model
