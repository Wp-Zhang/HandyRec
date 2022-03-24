---
title: models.rank.YouTubeRankDNN
permalink: docs/en/models-rank-YouTubeRankDNN
key: models-rank-YouTubeRankDNN
---
<!-- markdownlint-disable -->

<a href="..\handyrec\models\rank\YouTubeRankDNN.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.rank.YouTubeRankDNN`





---

<a href="..\handyrec\models\rank\YouTubeRankDNN.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `YouTubeRankDNN`

```python
YouTubeRankDNN(
    user_features: List[Any],
    item_features: List[Any],
    dnn_hidden_units: Tuple[int] = (64, 32),
    dnn_activation: str = 'relu',
    l2_dnn: float = 0,
    l2_emb: float = 1e-06,
    dnn_dropout: float = 0,
    seed: int = 2022
) → Model
```

Implementation of YoutubeDNN rank model 



**Args:**
 
 - <b>`user_features`</b> (List[Any]):  user feature list 
 - <b>`item_features`</b> (List[Any]):  item feature list 
 - <b>`dnn_hidden_units`</b> (Tuple[int], optional):  DNN structure. Defaults to (64, 32). 
 - <b>`dnn_activation`</b> (str, optional):  DNN activation function. Defaults to "relu". 
 - <b>`l2_dnn`</b> (float, optional):  DNN l2 regularization param. Defaults to 0. 
 - <b>`l2_emb`</b> (float, optional):  embedding l2 regularization param. Defaults to 1e-6. 
 - <b>`dnn_dropout`</b> (float, optional):  DNN dropout ratio. Defaults to 0. 
 - <b>`seed`</b> (int, optional):  random seed of dropout. Defaults to 2022. 



**Raises:**
 
 - <b>`ValueError`</b>:  length of `user_features` should be larger than 0 



**Returns:**
 
 - <b>`Model`</b>:  YouTubeDNN Match Model 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
