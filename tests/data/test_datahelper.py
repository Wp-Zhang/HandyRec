from handyrec.data import DataHelper
import pytest


def test_datahelper_abstract_func():
    DataHelper.__abstractmethods__ = set()

    class CustomDataHelper(DataHelper):
        def __init__(self, data_dir: str):
            super().__init__(data_dir)

    data_helper = CustomDataHelper("test_dir")

    data1 = data_helper.load_data()
    data2 = data_helper.preprocess_data()
    data3 = data_helper.get_clean_data()

    assert data_helper.base == "test_dir"
    assert data1 is None
    assert data2 is None
    assert data3 is None
