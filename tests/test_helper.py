from handyrec.data import PointWiseDataset, SequenceWiseDataset, PairWiseDataset
from handyrec.data.movielens import MovielensDataHelper
from handyrec.data.utils import gen_sequence
import numpy as np


def get_ml_test_data():
    dh = MovielensDataHelper("tests/ml-1m-test/")
    data = dh.get_clean_data(
        sparse_features=["gender", "occupation", "zip", "age", "year"]
    )
    data["inter"]["hist_movie"] = gen_sequence(data["inter"], "user_id", "movie_id", 2)

    return data


def get_pointwise_dataset(task="retrieval"):
    data = get_ml_test_data()

    user_features = ["user_id", "gender", "occupation", "zip", "age"]
    item_features = ["movie_id", "year", "genres"]
    inter_features = ["hist_movie"]

    dataset = PointWiseDataset(
        "Dataset",
        task=task,
        data=data,
        uid_name="user_id",
        iid_name="movie_id",
        inter_name="interact",
        time_name="timestamp",
        threshold=4,
    )

    dataset.train_test_split(1)
    if task == "ranking":
        dataset.negative_sampling(1)
    dataset.train_valid_split(0.1)
    if task == "ranking":
        uids = dataset.test_inter["user_id"].values
        candidates = {u: np.random.randint(1, 10, size=(5,)) for u in uids}
        dataset.gen_dataset(user_features, item_features, inter_features, candidates)
    else:
        dataset.gen_dataset(user_features, item_features, inter_features, shuffle=False)

    return dataset


def get_sequencewise_dataset():
    data = get_ml_test_data()

    user_features = ["user_id", "gender", "occupation", "zip", "age"]
    item_features = ["movie_id", "year", "genres"]
    inter_features = ["hist_movie"]

    dataset = SequenceWiseDataset(
        "RankingDataset",
        task="ranking",
        data=data,
        uid_name="user_id",
        iid_name="movie_id",
        inter_name="interact",
        time_name="timestamp",
        label_name="label",
        seq_name="hist_movie",
        neg_seq_name="neg_hist_movie",
        threshold=4,
    )

    dataset.train_test_split(1)
    dataset.negative_sampling(1)
    dataset.train_valid_split(0.1)
    uids = dataset.test_inter["user_id"].values
    candidates = {u: np.random.randint(1, 10, size=(5,)) for u in uids}
    dataset.gen_dataset(user_features, item_features, inter_features, candidates)

    return dataset


def get_pairwise_dataset():
    data = get_ml_test_data()

    user_features = ["user_id", "gender", "occupation", "zip", "age"]
    item_features = ["movie_id", "year", "genres"]
    inter_features = ["hist_movie"]

    dataset = PairWiseDataset(
        "RankingDataset",
        task="ranking",
        data=data,
        uid_name="user_id",
        iid_name="movie_id",
        inter_name="interact",
        time_name="timestamp",
        neg_iid_name="neg_movie_id",
        threshold=4,
    )

    dataset.train_test_split(1)
    dataset.negative_sampling(1)
    dataset.train_valid_split(0.1)
    uids = dataset.test_inter["user_id"].values
    candidates = {u: np.random.randint(1, 10, size=(5,)) for u in uids}
    dataset.gen_dataset(user_features, item_features, inter_features, candidates)

    return dataset
