---
title: layers.utils
permalink: docs/en/layers-utils
key: layers-utils
---
<!-- markdownlint-disable -->

<a href="..\handyrec\layers\utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `layers.utils`





---

<a href="..\handyrec\layers\utils.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `construct_input_layers`

```python
construct_input_layers(
    features: Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]
) → OrderedDict[str, Input]
```

Generate input layers repectively for each feature 

DenseFeature(FEAT_NAME) -> Input(FEAT_NAME) SparseFeature(FEAT_NAME) -> Input(FEAT_NAME) SparseSeqFeature(FEAT_NAME) -> Input(sparse_feat.FEAT_NAME) base sparse feature  Input(FEAT_NAME)             sparse feature index seq  Input(FEAT_NAME_len)         sparse feature seq length 



**Args:**
 
 - <b>`features`</b> (Union[List[DenseFeature], List[SparseFeature], List[SparseSeqFeature]]):  feature list 



**Returns:**
 
 - <b>`Dict[str, InputLayer]`</b>:  dictionary of input layers, {name: input layer} 


---

<a href="..\handyrec\layers\utils.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `construct_embedding_layers`

```python
construct_embedding_layers(
    sparse_features: Union[List[SparseFeature], List[SparseSeqFeature]],
    l2_reg: float
) → OrderedDict[str, Embedding]
```

Generate embedding layers for sparse features 



**Args:**
 
 - <b>`sparse_features`</b> (Union[List[SparseFeature], List[SparseSeqFeature]]):  sparse feature list 
 - <b>`l2_reg`</b> (float):  l2 regularization parameter 



**Returns:**
 
 - <b>`Dict[str, Embedding]`</b>:  dictionary of embedding layers, {name: embedding layer} 


---

<a href="..\handyrec\layers\utils.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `concatenate`

```python
concatenate(inputs, axis: int = -1)
```

Concatenate list of input, handle the case when len(inputs)=1 



**Args:**
 
 - <b>`inputs `</b>:  list of input 
 - <b>`axis`</b> (int, optional):  concatenate axis. Defaults to -1. 
 - <b>`# mask (bool, optional)`</b>:  whether to keep masks of input tensors. Defaults to Ture. 



**Returns:**
 
 - <b>`_type_`</b>:  concatenated input 


---

<a href="..\handyrec\layers\utils.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `concat_inputs`

```python
concat_inputs(
    dense_inputs: List,
    embd_inputs: List,
    axis: int = -1,
    keepdims: bool = False
)
```

Concatenate dense features and embedding of sparse features together 



**Args:**
 
 - <b>`dense_inputs`</b> (List):  dense features 
 - <b>`embd_inputs`</b> (List):  embedding of sparse features 
 - <b>`axis`</b> (int, optional):  concatenate axis. Deafults to `-1` 
 - <b>`keepdims`</b> (bool, optional):  whether to flatten all inputs before concatenating. Defaults to `False` 
 - <b>`# mask (bool, optional)`</b>:  whether to keep masks of input tensors. Defaults to Ture. 


---

<a href="..\handyrec\layers\utils.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `sampledsoftmaxloss`

```python
sampledsoftmaxloss(y_true, y_pred)
```

Helper function for calculating sampled softmax loss 



**Args:**
 
 - <b>`y_true `</b>:  label 
 - <b>`y_pred `</b>:  prediction 



**Returns:**
 
 - <b>`_type_`</b>:  loss 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
