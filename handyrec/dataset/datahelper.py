from typing import Dict, List


class DataHelper:
    def __init__(self, data_dir: str):
        self.base = data_dir  # diectory to load raw data and save generated dataset

    def load_data(self, *args, **kwargs) -> Dict:
        """Load raw dataset

        Returns:
            Dict: a dictionary with three keys: [`user`, `item`, `interact`]
        """
        pass

    def preprocess_data(self, data: Dict, *args, **kwargs) -> Dict:
        """Preprocess raw data

        Args:
            data (Dict): raw dataset dictionary

        Returns:
            dict: a dictionary of preprocessed data with three keys: [`user`, `item`, `interact`]
        """
        pass

        return data

    def get_clean_data(self, *args, **kwargs) -> Dict:
        """Load raw data and preprocess

        Returns:
            Dict: a dictionary of preprocessed data with three keys: [`user`, `item`, `interact`]
        """
        data = self.load_data()
        data = self.preprocess_data(data, *args, **kwargs)

        return data

    def gen_data_set(self, data: Dict, *args, **kwargs):
        """Generate and save dataset

        Args:
            data (Dict): a dictionary of preprocessed data with three keys: [`user`, `item`, `interact`]
        """
        pass

    def load_dataset(self, *args, **kwargs) -> Dict:
        """Load dataset into a dictionary

        Returns:
            Dict: a dictionary with feature names as keys
        """
        pass

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
