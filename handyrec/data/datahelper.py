from typing import Dict
from abc import ABC, abstractmethod


class DataHelper(ABC):
    """Abstract class for data loading and preprocesing

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
