from typing import Dict, List
from abc import ABC, abstractmethod


class DataHelper(ABC):
    """Abstract class for data loading, preprocesing, and dataset generating"""

    def __init__(self, data_dir: str):
        self.base = data_dir  # diectory to load raw data and save generated dataset

    @abstractmethod
    def load_data(self) -> Dict:
        """Load raw dataset

        Returns:
            Dict: a data dict with three keys: [`user`, `item`, `interact`]
        """

    @abstractmethod
    def preprocess_data(self) -> Dict:
        """Preprocess raw data

        Returns:
            dict: a data dict with three keys: [`user`, `item`, `interact`]
        """

    @abstractmethod
    def get_clean_data(self) -> Dict:
        """Load raw data and preprocess

        Returns:
            dict: a data dict with three keys: [`user`, `item`, `interact`]
        """

    @abstractmethod
    def gen_data_set(self):
        """Generate and save dataset"""

    @abstractmethod
    def load_dataset(self) -> Dict:
        """Load dataset into a dictionary

        Returns:
            Dict: a data dict with feature names as keys
        """

    def get_feature_dim(
        self,
        data: Dict,
        user_features: List[str],
        item_features: List[str],
        interact_features: List[str],
    ) -> Dict:
        """Generate a dictionary containing feature dimensions

        Args:
            data (Dict): dataset dictionary
            user_features (List[str]): user feature list
            item_features (List[str]): item feature list
            interact_features (List[str]): user-item interaction feature list

        Returns:
            Dict: feature dimension dict. {feature: dimension}
        """
        feature_dim = {}
        for feat in user_features:
            feature_dim[feat] = data["user"][feat].max() + 1
        for feat in item_features:
            feature_dim[feat] = data["item"][feat].max() + 1
        for feat in interact_features:
            feature_dim[feat] = data["interact"][feat].max() + 1

        return feature_dim
