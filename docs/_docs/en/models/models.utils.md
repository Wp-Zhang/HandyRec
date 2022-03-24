---
title: models.utils
permalink: docs/en/models-utils
key: models-utils
---
<!-- markdownlint-disable -->

<a href="..\handyrec\models\utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models.utils`





---

<a href="..\handyrec\models\utils.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `search_embedding`

```python
search_embedding(
    embd_dim: int,
    item_embd: ndarray,
    user_embd: ndarray,
    item_list: ndarray,
    n: int,
    gpu: bool = False
) → ndarray
```

Search top n similar item embeddings for each user embedding 



**Args:**
 
 - <b>`embd_dim`</b> (int):  embedding dimension 
 - <b>`item_embd`</b> (np.ndarray):  item embedding 
 - <b>`user_embd`</b> (np.ndarray):  user embedding 
 - <b>`user_ids`</b> (Iterable):  list of target users 
 - <b>`item_list`</b> (np.ndarray):  full item numpy array, has same length with `item_embd` 
 - <b>`n`</b> (int):  number of candidate items for each user 
 - <b>`gpu`</b> (bool, optional):  use gpu to search. Defaults to False. 



**Returns:**
 
 - <b>`np.array`</b>:  search result. (NUM_USERS x n) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
