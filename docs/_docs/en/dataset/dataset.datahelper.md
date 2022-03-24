---
title: dataset.datahelper
permalink: docs/en/dataset-datahelper
key: dataset-datahelper
---
<!-- markdownlint-disable -->

<a href="..\handyrec\dataset\datahelper.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dataset.datahelper`






---

<a href="..\handyrec\dataset\datahelper.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DataHelper`
Abstract class for data loading, preprocesing, and dataset generating 

<a href="..\handyrec\dataset\datahelper.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(data_dir: str)
```








---

<a href="..\handyrec\dataset\datahelper.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `gen_dataset`

```python
gen_dataset()
```

Generate and save dataset 

---

<a href="..\handyrec\dataset\datahelper.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_clean_data`

```python
get_clean_data() → Dict
```

Load raw data and preprocess 



**Returns:**
 
 - <b>`dict`</b>:  a data dict with three keys: [`user`, `item`, `interact`] 

---

<a href="..\handyrec\dataset\datahelper.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_feature_dim`

```python
get_feature_dim(
    data: Dict,
    user_features: List[str],
    item_features: List[str],
    interact_features: List[str]
) → Dict
```

Generate a dictionary containing feature dimensions 



**Args:**
 
 - <b>`data`</b> (Dict):  dataset dictionary 
 - <b>`user_features`</b> (List[str]):  user feature list 
 - <b>`item_features`</b> (List[str]):  item feature list 
 - <b>`interact_features`</b> (List[str]):  user-item interaction feature list 



**Returns:**
 
 - <b>`Dict`</b>:  feature dimension dict. {feature: dimension} 

---

<a href="..\handyrec\dataset\datahelper.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_data`

```python
load_data() → Dict
```

Load raw dataset 



**Returns:**
 
 - <b>`Dict`</b>:  a data dict with three keys: [`user`, `item`, `interact`] 

---

<a href="..\handyrec\dataset\datahelper.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dataset`

```python
load_dataset() → Dict
```

Load dataset into a dictionary 



**Returns:**
 
 - <b>`Dict`</b>:  a data dict with feature names as keys 

---

<a href="..\handyrec\dataset\datahelper.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `preprocess_data`

```python
preprocess_data() → Dict
```

Preprocess raw data 



**Returns:**
 
 - <b>`dict`</b>:  a data dict with three keys: [`user`, `item`, `interact`] 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
