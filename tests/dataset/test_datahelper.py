from handyrec.dataset import DataHelper
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
    data4 = data_helper.gen_dataset()
    data5 = data_helper.load_dataset()

    assert data_helper.base == "test_dir"
    assert data1 is None
    assert data2 is None
    assert data3 is None
    assert data4 is None
    assert data5 is None


@pytest.mark.parametrize(
    "user_features, item_features, interact_features, expected",
    [
        (["a"], ["b"], ["d"], {"a": 2, "b": 3, "d": 5}),
        (["a"], ["b", "c"], [], {"a": 2, "b": 3, "c": 4}),
    ],
)
def test_datahelper_get_feature_dim(
    user_features, item_features, interact_features, expected
):
    DataHelper.__abstractmethods__ = set()

    class CustomDataHelper(DataHelper):
        def __init__(self, data_dir: str):
            super().__init__(data_dir)

    data_helper = CustomDataHelper("test_dir")
    fake_data = {
        "user": {"a": [1]},
        "item": {"b": [1, 2], "c": [3]},
        "interact": {"d": [4]},
    }
    assert expected == data_helper.get_feature_dim(
        fake_data, user_features, item_features, interact_features
    )


def test_datahelper_get_feature_fail():
    DataHelper.__abstractmethods__ = set()

    class CustomDataHelper(DataHelper):
        def __init__(self, data_dir: str):
            super().__init__(data_dir)

    data_helper = CustomDataHelper("test_dir")
    fake_data = {
        "user": {"a": [1]},
        "interact": {},
    }
    with pytest.raises(Exception):
        data_helper.get_feature_dim(fake_data, ["a"], [], [])
