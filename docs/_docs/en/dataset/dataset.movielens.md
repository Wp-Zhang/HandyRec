---
title: dataset.movielens
permalink: docs/en/dataset-movielens
key: dataset-movielens
---
<!-- markdownlint-disable -->

<a href="..\handyrec\dataset\movielens.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dataset.movielens`






---

<a href="..\handyrec\dataset\movielens.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MovielensDataHelper`
base class for DataHelper for movielens dataset 

<a href="..\handyrec\dataset\movielens.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(data_dir: str, sub_dir_name: str)
```








---

<a href="..\handyrec\dataset\movielens.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_clean_data`

```python
get_clean_data(sparse_features: List[str]) → Dict
```

Load raw data and preprocess 



**Args:**
 
 - <b>`sparse_features`</b> (List[str]):  sparse feature list to be label encoded 



**Returns:**
 
 - <b>`Dict`</b>:  a dictionary of preprocessed data with three keys: [`user`, `item`, `interact`] 

---

<a href="..\handyrec\dataset\movielens.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_data`

```python
load_data() → Dict
```

Load original raw data 



**Returns:**
 
 - <b>`dict`</b>:  raw data dictionary 

---

<a href="..\handyrec\dataset\movielens.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `preprocess_data`

```python
preprocess_data(data: dict, sparse_features: List[str]) → dict
```

Preprocess raw data 



**Args:**
 
 - <b>`data`</b> (dict):  data dictionary, keys: 'item', 'user', 'interact' 
 - <b>`sparse_features`</b> (List[str]):  sparse feature list to be label encoded 



**Returns:**
 
 - <b>`dict`</b>:  data dictionary 


---

<a href="..\handyrec\dataset\movielens.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MovieMatchDataHelper`




<a href="..\handyrec\dataset\movielens.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(data_dir: str)
```








---

<a href="..\handyrec\dataset\movielens.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `gen_dataset`

```python
gen_dataset(
    features: List[str],
    data: dict,
    seq_max_len: int = 20,
    min_rating: float = 0.35,
    n: int = 10
)
```

Generate train set and test set 



**Args:**
 
 - <b>`features`</b> (List[str]):  feature list 
 - <b>`data`</b> (dict):  data dictionary, keys: 'user', 'item', 'interact' 
 - <b>`seq_max_len`</b> (int, optional):  maximum history sequence length. Defaults to 20. 
 - <b>`min_rating`</b> (float, optional):  minimum interact for positive smaples. Defaults to 0.35. 
 - <b>`n`</b> (int, optional):  use the last n samples for each user to be the test set. Defaults to 10. 

---

<a href="..\handyrec\dataset\movielens.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_clean_data`

```python
get_clean_data(sparse_features: List[str]) → Dict
```

Load raw data and preprocess 



**Args:**
 
 - <b>`sparse_features`</b> (List[str]):  sparse feature list to be label encoded 



**Returns:**
 
 - <b>`Dict`</b>:  a dictionary of preprocessed data with three keys: [`user`, `item`, `interact`] 

---

<a href="..\handyrec\dataset\movielens.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_data`

```python
load_data() → Dict
```

Load original raw data 



**Returns:**
 
 - <b>`dict`</b>:  raw data dictionary 

---

<a href="..\handyrec\dataset\movielens.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dataset`

```python
load_dataset(user_feats: List[str], movie_feats: List[str]) → Tuple
```

Load saved dataset 



**Args:**
 
 - <b>`data_name`</b> (str):  version name of data used to generate dataset 
 - <b>`dataset_name`</b> (str):  version name of dataset 
 - <b>`user_feats`</b> (List[str]):  list of user features to be loaded 
 - <b>`movie_feats`</b> (List[str]):  list of movie features to be loaded 



**Returns:**
 
 - <b>`Tuple`</b>:  [train set, test set] 

---

<a href="..\handyrec\dataset\movielens.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `preprocess_data`

```python
preprocess_data(data: dict, sparse_features: List[str]) → dict
```

Preprocess raw data 



**Args:**
 
 - <b>`data`</b> (dict):  data dictionary, keys: 'item', 'user', 'interact' 
 - <b>`sparse_features`</b> (List[str]):  sparse feature list to be label encoded 



**Returns:**
 
 - <b>`dict`</b>:  data dictionary 


---

<a href="..\handyrec\dataset\movielens.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MovieRankDataHelper`




<a href="..\handyrec\dataset\movielens.py#L272"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(data_dir: str)
```








---

<a href="..\handyrec\dataset\movielens.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `gen_dataset`

```python
gen_dataset(
    features: List[str],
    data: dict,
    test_id: dict,
    seq_max_len: int = 20,
    negnum: int = 0,
    min_rating: float = 0.35,
    n: int = 10
)
```

Generate train set and test set 



**Args:**
 
 - <b>`features`</b> (List[str]):  feature list 
 - <b>`data`</b> (dict):  data dictionary, keys: 'user', 'item', 'interact' 
 - <b>`test_id`</b> (dict):  test id dictionary, {user_id: movie_id}. Note: each user should have the same number of movie_ids 
 - <b>`seq_max_len`</b> (int, optional):  maximum history sequence length. Defaults to 20. 
 - <b>`negnum`</b> (int, optional):  number of negative samples. Defaults to 0. 
 - <b>`min_rating`</b> (float, optional):  minimum interact for positive smaples. Defaults to 0.35. 
 - <b>`n`</b> (int, optional):  use the last n samples for each user to be the test set. Defaults to 10. 

---

<a href="..\handyrec\dataset\movielens.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_clean_data`

```python
get_clean_data(sparse_features: List[str]) → Dict
```

Load raw data and preprocess 



**Args:**
 
 - <b>`sparse_features`</b> (List[str]):  sparse feature list to be label encoded 



**Returns:**
 
 - <b>`Dict`</b>:  a dictionary of preprocessed data with three keys: [`user`, `item`, `interact`] 

---

<a href="..\handyrec\dataset\movielens.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_data`

```python
load_data() → Dict
```

Load original raw data 



**Returns:**
 
 - <b>`dict`</b>:  raw data dictionary 

---

<a href="..\handyrec\dataset\movielens.py#L428"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dataset`

```python
load_dataset(user_feats: List[str], movie_feats: List[str]) → Tuple
```

Load saved dataset 



**Args:**
 
 - <b>`data_name`</b> (str):  version name of data used to generate dataset 
 - <b>`dataset_name`</b> (str):  version name of dataset 
 - <b>`user_feats`</b> (List[str]):  list of user features to be loaded 
 - <b>`movie_feats`</b> (List[str]):  list of movie features to be loaded 



**Returns:**
 
 - <b>`Tuple`</b>:  [train set, test set] 

---

<a href="..\handyrec\dataset\movielens.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `preprocess_data`

```python
preprocess_data(data: dict, sparse_features: List[str]) → dict
```

Preprocess raw data 



**Args:**
 
 - <b>`data`</b> (dict):  data dictionary, keys: 'item', 'user', 'interact' 
 - <b>`sparse_features`</b> (List[str]):  sparse feature list to be label encoded 



**Returns:**
 
 - <b>`dict`</b>:  data dictionary 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
