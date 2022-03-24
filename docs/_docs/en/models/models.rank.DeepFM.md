---
title: models.rank.DeepFM
permalink: docs/en/models-rank-DeepFM
key: models-rank-DeepFM
---
<!-- markdownlint-disable -->

<a href="..\handyrec\models\rank\DeepFM.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.rank.DeepFM`





---

<a href="..\handyrec\models\rank\DeepFM.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `DeepFM`

```python
DeepFM(
    fm_features: List[Any],
    dnn_features: List[Any],
    dnn_hidden_units: Tuple[int] = (64, 32, 1),
    dnn_activation: str = 'relu',
    dnn_dropout: float = 0,
    dnn_bn: bool = False,
    l2_dnn: float = 0,
    l2_emb: float = 1e-06,
    task: str = 'binary',
    seed: int = 2022
)
```

Implementation of DeepFM 



**Args:**
 
 - <b>`fm_features`</b> (List[Any]):  input feature list for FM 
 - <b>`dnn_features`</b> (List[Any]):  input feature list for DNN 
 - <b>`dnn_hidden_units`</b> (Tuple[int], optional):  DNN structure. Defaults to (64, 32). 
 - <b>`dnn_activation`</b> (str, optional):  DNN activation function. Defaults to "relu". 
 - <b>`dnn_dropout`</b> (float, optional):  DNN dropout ratio. Defaults to 0. 
 - <b>`dnn_bn`</b> (bool, optional):  whether to use batch normalization. Defaults to False. 
 - <b>`l2_dnn`</b> (float, optional):  DNN l2 regularization param. Defaults to 0. 
 - <b>`l2_emb`</b> (float, optional):  embedding l2 regularization param. Defaults to 1e-6. 
 - <b>`task`</b> (str, optional):  model task, should be `binary` or `regression`. Defaults to `binary` 
 - <b>`seed`</b> (int, optional):  random seed of dropout. Defaults to 2022. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
