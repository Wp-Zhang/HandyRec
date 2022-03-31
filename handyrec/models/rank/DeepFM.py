"""Implementation of DeepFM
"""
from typing import Tuple
import warnings
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from handyrec.features import FeatureGroup
from handyrec.layers import DNN, FM
from handyrec.layers.utils import concat


def DeepFM(
    fm_feature_group: FeatureGroup,
    dnn_feature_group: FeatureGroup,
    dnn_hidden_units: Tuple[int] = (64, 32, 1),
    dnn_activation: str = "relu",
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    task: str = "binary",
    seed: int = 2022,
) -> Model:
    """Implementation of DeepFM

    Parameters
    ----------
    fm_feature_group : FeatureGroup
        FM feature group.
    dnn_feature_group : FeatureGroup
        DNN feature group.
    dnn_hidden_units : Tuple[int], optional
        DNN structure, by default ``(64, 32, 1)``.
    dnn_activation : str, optional
        DNN activation function, by default ``"relu"``.
    dnn_dropout : float, optional
        DNN dropout ratio, by default ``0``.
    dnn_bn : bool, optional
        Whether to use batch normalization, by default ``False``.
    l2_dnn : float, optional
        DNN l2 regularization param, by default ``0``.
    task : str, optional
        Model task, should be ``"binary"`` or ``"regression"``, by default ``"binary"``.
    seed : int, optional
        Random seed of dropout, by default ``2022``.

    Returns
    -------
    Model
        A DeepFM model.

    Raises
    ------
    ValueError
        If the size of DNN's last layer is not 1.

    References
    ----------
    .. [1] Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network
        for CTR prediction." arXiv preprint arXiv:1703.04247 (2017).
    """
    if dnn_hidden_units[-1] != 1:
        raise ValueError("Output size of dnn should be 1")

    fm_dense, fm_sparse = fm_feature_group.embedding_lookup(pool_method="mean")
    dnn_dense, dnn_sparse = dnn_feature_group.embedding_lookup(pool_method="mean")

    if len(fm_dense) > 0:
        warnings.warn(
            "FM currently doesn't support dense featrue, they will be ignored"
        )
    # * Concat input layers -> DNN, FM
    dnn_input = concat(fm_dense + dnn_dense, fm_sparse + dnn_sparse)
    fm_input = concat([], fm_sparse + dnn_sparse, axis=1, keepdims=True)

    dnn_output = DNN(
        hidden_units=dnn_hidden_units,
        activation=dnn_activation,
        l2_reg=l2_dnn,
        dropout_rate=dnn_dropout,
        use_bn=dnn_bn,
        output_activation="linear",
        seed=seed,
        name="Deep_Part",
    )(dnn_input)
    fm_output = FM(name="FM_Part")(fm_input)

    # * Output
    output = dnn_output + fm_output
    if task == "binary":
        output = Activation("sigmoid")(output)

    # * Construct model
    inputs = list(fm_feature_group.feat_pool.input_layers.values())
    model = Model(inputs=inputs, outputs=output)

    return model
