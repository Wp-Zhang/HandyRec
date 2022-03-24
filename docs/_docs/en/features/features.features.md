---
title: features.features
permalink: docs/en/features-features
key: features-features
---
<!-- markdownlint-disable -->

<a href="..\handyrec\features\features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `features.features`






---

<a href="..\handyrec\features\features.py#L1"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Feature`
Base class for different types of features 

<a href="..\handyrec\features\features.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```









---

<a href="..\handyrec\features\features.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DenseFeature`
Dense Feature class 

<a href="..\handyrec\features\features.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(name: str, dtype: str = 'int32')
```









---

<a href="..\handyrec\features\features.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SparseFeature`
Sparse feature class 

<a href="..\handyrec\features\features.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(name: str, vocab_size: int, embedding_dim: int, dtype: str = 'int32')
```



**Args:**
 
 - <b>`name`</b> (str):  Name of feature, each feature should have a distinct name. 
 - <b>`vocab_size`</b> (int):  Vocabulary size. 
 - <b>`embedding_dim`</b> (int):  Embedding dimension. 
 - <b>`dtype`</b> (str, optional):  Data type. Defaults to 'int32'. 





---

<a href="..\handyrec\features\features.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SparseSeqFeature`
Sparse sequence feature, e.g. item_id sequence 

<a href="..\handyrec\features\features.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(sparse_feat: SparseFeature, name: str, seq_len: int)
```



**Args:**
 
 - <b>`sparse_feat`</b> (SparseFeature):  sparse sequence feature 
 - <b>`name`</b> (str):  feature name 
 - <b>`seq_len`</b> (int):  sequence length 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
