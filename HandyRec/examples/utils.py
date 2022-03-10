import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Tuple
import numpy as np
from typing import List
import gc
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataProcessor:
    def __init__(self, data_dir: str):
        self.base = data_dir  # data diectory

    def _load_raw_data(self) -> dict:
        """Load original raw data

        Returns:
            dict: raw data dictionary
        """
        unames = ["user_id", "gender", "age", "occupation", "zip"]
        user = pd.read_csv(self.base + "users.dat", sep="::", header=None, names=unames)
        rnames = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            self.base + "ratings.dat", sep="::", header=None, names=rnames
        )
        mnames = ["movie_id", "title", "genres"]
        movies = pd.read_csv(
            self.base + "movies.dat", sep="::", header=None, names=mnames
        )

        return {"item": movies, "user": user, "rating": ratings}

    def _transform_feats(self, data: dict, sparse_features: List[str]) -> dict:
        """Label encode sparse features

        Args:
            data (dict): data dictionary, keys: 'item', 'user', 'rating'
            sparse_features (List[str]): sparse feature list

        Returns:
            dict: data dictionary
        """
        user = data["user"]
        item = data["item"]

        # Users
        for feat in tqdm(
            [f for f in sparse_features if f in user.columns],
            "Encode User Sparse Feats",
        ):
            lbe = LabelEncoder()
            user[feat] = lbe.fit_transform(user[feat].astype(str)) + 1
            user[feat] = user[feat].astype("int32")

        # Movies
        for feat in tqdm(
            [f for f in sparse_features if f in item.columns],
            "Encode Item Sparse Feats",
        ):
            lbe = LabelEncoder()
            item[feat] = lbe.fit_transform(item[feat].astype(str)) + 1
            item[feat] = item[feat].astype("int32")

        data["user"] = user
        data["item"] = item

        return data

    def preprocess_data(self, lbd_feats: List[str]) -> dict:
        """Preprocess raw data

        Args:
            lbd_feats (List[str]): categorical features to be label encoded

        Returns:
            dict: preprocessed data
        """
        data = self._load_raw_data()
        data = self._transform_feats(data, lbd_feats)

        return data

    def gen_data_set(
        self,
        features: List[str],
        data: dict,
        seq_max_len: int = 20,
        negnum: int = 0,
        min_rating: float = 0.35,
        n: int = 10,
    ):
        """Generate train set and test set

        Args:
            features (List[str]): feature list
            data (dict): data dictionary, keys: 'user', 'item', 'rating'
            seq_max_len (int, optional): maximum history sequence length. Defaults to 20.
            negnum (int, optional): number of negative samples. Defaults to 0.
            min_rating (float, optional): minimum rating for positive smaples. Defaults to 0.35.
            n (int, optional): use the last n samples for each user to be the test set. Defaults to 10.

        """

        data["rating"].sort_values("timestamp", inplace=True)
        df = data["rating"]

        # Split train set and test set
        item_ids = set(data["item"]["movie_id"].values)

        # Calculate number of rows of daraset to fasten the dataset generating process
        df = df[df["rating"] >= min_rating]
        counter = df[["user_id", "movie_id"]].groupby("user_id", as_index=False).count()
        counter = counter[counter["movie_id"] > n]
        df = df[df["user_id"].isin(counter["user_id"].values)]
        train_rows = ((counter["movie_id"] - n) * (negnum + 1)).sum()
        test_rows = counter.shape[0]

        # Generate rows
        # train_set format: [uid, moiveID, label, history_seq_len, history_seq]
        # test_set format: [uid, moiveIDs, history_seq_len, history_seq]
        train_set = np.zeros((train_rows, 5), dtype=object)
        test_set = np.zeros((test_rows, 4), dtype=object)

        p, q = 0, 0
        for uid, hist in tqdm(df.groupby("user_id"), "Generate train set"):
            pos_list = hist["movie_id"].tolist()
            if negnum > 0:
                candidate_set = list(item_ids - set(pos_list))  # Negative samples
                negs = np.random.choice(
                    candidate_set, size=len(pos_list) * negnum, replace=True
                )
            for i in range(len(pos_list) - n):
                hist = pos_list[: i + 1]
                # Positive sample
                train_set[p] = [uid, pos_list[i], 1, len(hist), hist[::-1]]
                p += 1
                # Negative smaples
                for j in range(negnum):
                    train_set[p] = [uid, negs[i * negnum + j], 0, len(hist), hist[::-1]]
                    p += 1
            i = len(pos_list) - n
            test_set[q] = [uid, pos_list[i:], i, pos_list[:i][::-1]]
            q += 1
        # test_set = test_set[np.isin(test_set[:,0], test_users)]

        np.random.seed(2022)
        np.random.shuffle(train_set)
        np.random.shuffle(test_set)

        user = data["user"]
        item = data["item"]
        user = user.set_index("user_id")
        item = item.set_index("movie_id")

        train_uid = train_set[:, 0].astype(np.int)
        train_iid = train_set[:, 1].astype(np.int)
        train_label = train_set[:, 2].astype(np.int)
        hist_seq_len = train_set[:, 3].astype(np.int)
        hist_seq = train_set[:, 4].tolist()
        hist_seq_pad = pad_sequences(
            hist_seq, maxlen=seq_max_len, padding="post", truncating="post", value=0
        )
        np.save(open(self.base + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.base + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.base + "train_label.npy", "wb"), train_label)
        np.save(open(self.base + "train_hist_movie_id_len.npy", "wb"), hist_seq_len)
        np.save(open(self.base + "train_hist_movie_id.npy", "wb"), hist_seq_pad)

        test_uid = test_set[:, 0].astype(np.int)
        test_label = np.array(test_set[:, 1].tolist()).astype(np.int)
        hist_seq_len = test_set[:, 2].astype(np.int)
        hist_seq = test_set[:, 3].tolist()
        hist_seq_pad = pad_sequences(
            hist_seq, maxlen=seq_max_len, padding="post", truncating="post", value=0
        )
        np.save(open(self.base + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.base + "test_label.npy", "wb"), test_label)
        np.save(open(self.base + "test_hist_movie_id_len.npy", "wb"), hist_seq_len)
        np.save(open(self.base + "test_hist_movie_id.npy", "wb"), hist_seq_pad)

        del train_set, test_set, hist_seq, hist_seq_pad
        gc.collect()

        for key in tqdm([x for x in user.columns if x in features and x != "user_id"]):
            train_tmp_array = user[key].loc[train_uid].values
            test_tmp_array = user[key].loc[test_uid].values
            np.save(open(self.base + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.base + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

        del train_uid, user
        gc.collect()

        for key in tqdm([x for x in item.columns if x in features and x != "movie_id"]):
            train_tmp_array = item[key].loc[train_iid].values
            np.save(open(self.base + "train_" + key + ".npy", "wb"), train_tmp_array)
            del train_tmp_array
            gc.collect()

    def load_dataset(
        self,
        user_feats: List[str],
        movie_feats: List[str],
    ) -> Tuple:
        """Load saved dataset

        Args:
            data_name (str): version name of data used to generate dataset
            dataset_name (str): version name of dataset
            user_feats (List[str]): list of user features to be loaded
            movie_feats (List[str]): list of movie features to be loaded

        Returns:
            Tuple: [train set, test set]
        """
        train_set = {}
        test_set = {}

        for feat in tqdm(
            user_feats + ["hist_movie_id", "hist_movie_id_len"], "Load user Features"
        ):
            train_set[feat] = np.load(
                open(self.base + "train_" + feat + ".npy", "rb"), allow_pickle=True
            )
            test_set[feat] = np.load(
                open(self.base + "test_" + feat + ".npy", "rb"), allow_pickle=True
            )

        for feat in tqdm(movie_feats, "Load movie Features"):
            train_set[feat] = np.load(
                open(self.base + "train_" + feat + ".npy", "rb"), allow_pickle=True
            )

        train_label = np.load(
            open(self.base + "train_label.npy", "rb"), allow_pickle=True
        )
        test_label = np.load(
            open(self.base + "test_label.npy", "rb"), allow_pickle=True
        )

        return train_set, train_label, test_set, test_label


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    return np.mean(
        [apk(a, p, k) for a, p in zip(actual, predicted) if a]
    )  # CHANGES: ignore null actual (variable=a)
