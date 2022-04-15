from typing import Dict, List
from abc import ABC, abstractmethod
import numpy as np


class DataHelper(ABC):
    """Abstract class for data loading, preprocesing, and dataset generating

    Attributes
    ----------
    base : str
        Diectory to load raw data and save generated dataset.
    """

    def __init__(self, data_dir: str):
        self.base = data_dir

    @abstractmethod
    def load_data(self) -> Dict:
        """Load raw dataset into a dictionary.

        Returns
        -------
        Dict
            A data dict with three keys: ``user``, ``item``, and ``inter``.
        """

    @abstractmethod
    def preprocess_data(self) -> Dict:
        """Preprocess raw data.

        Returns
        -------
        Dict
            A data dict with three keys: ``user``, ``item``, and ``inter``.
        """

    @abstractmethod
    def get_clean_data(self) -> Dict:
        """Load raw data and preprocess.

        Returns
        -------
        Dict
            A data dict with three keys: ``user``, ``item``, and ``inter``.
        """

    @classmethod
    def get_feature_dim(
        cls,
        data: Dict,
        user_features: List[str],
        item_features: List[str],
        interact_features: List[str],
    ) -> Dict[str, int]:
        """Generate a dictionary containing feature dimension info.

        Parameters
        ----------
        data : Dict
            Dataset dictionary.
        user_features : List[str]
            User feature list.
        item_features : List[str]
            Item feature list.
        interact_features : List[str]
            User-item interaction feature list.

        Returns
        -------
        Dict[str, int]
            A dictionary containing feature dimension info, ``{feature: dim}``.

        Raises
        ------
        KeyError
            If ``user``,``item``, or ``inter`` is not in ``data.keys()``
        """
        if len(set(["user", "item", "inter"]) & set(data.keys())) != 3:
            raise KeyError("`user`,`item`, and `inter` should be keys of data")

        feature_dim = {}
        for feat in user_features:
            try:
                feature_dim[feat] = np.max(data["user"][feat]) + 1
            except:
                pass
        for feat in item_features:
            try:
                feature_dim[feat] = np.max(data["item"][feat]) + 1
            except:
                pass
        for feat in interact_features:
            try:
                feature_dim[feat] = np.max(data["inter"][feat]) + 1
            except:
                pass

        return feature_dim
