import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from typing import Tuple, List
import numpy as np
import gc
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


class DataHelper:
    def __init__(self, data_dir: str):
        self.base = data_dir  # data diectory

    def _load_raw_data(self) -> dict:
        """Load original raw data

        Returns:
            dict: raw data dictionary
        """
        unames = ["user_id", "gender", "age", "occupation", "zip"]
        user = pd.read_csv(
            self.base + "users.dat",
            sep="::",
            header=None,
            names=unames,
            encoding="latin-1",
        )
        rnames = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            self.base + "ratings.dat",
            sep="::",
            header=None,
            names=rnames,
            encoding="latin-1",
        )
        mnames = ["movie_id", "title", "genres"]
        movies = pd.read_csv(
            self.base + "movies.dat",
            sep="::",
            header=None,
            names=mnames,
            encoding="latin-1",
        )
        genres = movies["genres"].str.get_dummies(sep="|")
        movies = pd.concat([movies[["movie_id", "title"]], genres], axis=1)

        return {"item": movies, "user": user, "rating": ratings}

    def _encode_feats(self, data: dict, sparse_features: List[str]) -> dict:
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
        data = self._encode_feats(data, lbd_feats)

        return data

    def gen_data_set(self):
        pass

    def load_dataset(self):
        pass


class MatchDataHelper(DataHelper):
    def __init__(self, data_dir: str):
        super(MatchDataHelper, self).__init__(data_dir)
        self.sub_dir = data_dir + "match/"
        if not os.path.exists(self.sub_dir):
            os.makedirs(self.sub_dir)

    def gen_data_set(
        self,
        features: List[str],
        data: dict,
        seq_max_len: int = 20,
        min_rating: float = 0.35,
        n: int = 10,
    ):
        """Generate train set and test set

        Args:
            features (List[str]): feature list
            data (dict): data dictionary, keys: 'user', 'item', 'rating'
            seq_max_len (int, optional): maximum history sequence length. Defaults to 20.
            min_rating (float, optional): minimum rating for positive smaples. Defaults to 0.35.
            n (int, optional): use the last n samples for each user to be the test set. Defaults to 10.

        """

        data["rating"].sort_values("timestamp", inplace=True)
        df = data["rating"]

        # Calculate number of rows of daraset to fasten the dataset generating process
        df = df[df["rating"] >= min_rating]
        counter = df[["user_id", "movie_id"]].groupby("user_id", as_index=False).count()
        counter = counter[counter["movie_id"] > n]
        df = df[df["user_id"].isin(counter["user_id"].values)]
        train_rows = (counter["movie_id"] - n).sum()
        test_rows = counter.shape[0]

        # Generate rows
        # train_set format: [uid, moiveID, sample_age, label, history_seq_len, history_seq]
        # test_set format: [uid, moiveIDs, sample_age(0), history_seq_len, history_seq]
        train_set = np.zeros((train_rows, 6), dtype=object)
        test_set = np.zeros((test_rows, 5), dtype=object)

        p, q = 0, 0
        for uid, hist in tqdm(df.groupby("user_id"), "Generate train set"):
            pos_list = hist["movie_id"].tolist()
            for i in range(len(pos_list) - n):
                seq = pos_list[:i]
                # Positive sample
                train_set[p] = [
                    uid,
                    pos_list[i],
                    len(pos_list) - n - 1 - i,
                    1,
                    len(seq),
                    seq[::-1],
                ]
                p += 1
            i = len(pos_list) - n
            test_set[q] = [uid, pos_list[i:], 0, i, pos_list[:i][::-1]]
            q += 1

        np.random.seed(2022)
        np.random.shuffle(train_set)
        np.random.shuffle(test_set)

        user = data["user"]
        item = data["item"]
        user = user.set_index("user_id")
        item = item.set_index("movie_id")

        train_uid = train_set[:, 0].astype(np.int)
        train_iid = train_set[:, 1].astype(np.int)
        train_age = train_set[:, 2].astype(np.int)
        train_label = train_set[:, 3].astype(np.int)
        hist_seq_len = train_set[:, 4].astype(np.int)
        hist_seq = train_set[:, 5].tolist()
        hist_seq_pad = pad_sequences(
            hist_seq, maxlen=seq_max_len, padding="post", truncating="post", value=0
        )
        np.save(open(self.sub_dir + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.sub_dir + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.sub_dir + "train_example_age.npy", "wb"), train_age)
        np.save(open(self.sub_dir + "train_label.npy", "wb"), train_label)
        np.save(open(self.sub_dir + "train_hist_movie_id_len.npy", "wb"), hist_seq_len)
        np.save(open(self.sub_dir + "train_hist_movie_id.npy", "wb"), hist_seq_pad)

        test_uid = test_set[:, 0].astype(np.int)
        test_label = np.array(test_set[:, 1].tolist()).astype(np.int)
        test_age = test_set[:, 2].astype(np.int)
        hist_seq_len = test_set[:, 3].astype(np.int)
        hist_seq = test_set[:, 4].tolist()
        hist_seq_pad = pad_sequences(
            hist_seq, maxlen=seq_max_len, padding="post", truncating="post", value=0
        )
        np.save(open(self.sub_dir + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.sub_dir + "test_example_age.npy", "wb"), test_age)
        np.save(open(self.sub_dir + "test_label.npy", "wb"), test_label)
        np.save(open(self.sub_dir + "test_hist_movie_id_len.npy", "wb"), hist_seq_len)
        np.save(open(self.sub_dir + "test_hist_movie_id.npy", "wb"), hist_seq_pad)

        del train_set, test_set, hist_seq, hist_seq_pad
        gc.collect()

        for key in tqdm([x for x in user.columns if x in features and x != "user_id"]):
            train_tmp_array = user[key].loc[train_uid].values
            test_tmp_array = user[key].loc[test_uid].values
            np.save(open(self.sub_dir + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.sub_dir + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

        del train_uid, user
        gc.collect()

        for key in tqdm([x for x in item.columns if x in features and x != "movie_id"]):
            train_tmp_array = item[key].loc[train_iid].values
            np.save(open(self.sub_dir + "train_" + key + ".npy", "wb"), train_tmp_array)
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
            user_feats + ["hist_movie_id", "hist_movie_id_len", "example_age"],
            "Load user Features",
        ):
            train_set[feat] = np.load(
                open(self.sub_dir + "train_" + feat + ".npy", "rb"), allow_pickle=True
            )
            test_set[feat] = np.load(
                open(self.sub_dir + "test_" + feat + ".npy", "rb"), allow_pickle=True
            )

        for feat in tqdm(movie_feats, "Load movie Features"):
            train_set[feat] = np.load(
                open(self.sub_dir + "train_" + feat + ".npy", "rb"), allow_pickle=True
            )

        train_label = np.load(
            open(self.sub_dir + "train_label.npy", "rb"), allow_pickle=True
        )
        test_label = np.load(
            open(self.sub_dir + "test_label.npy", "rb"), allow_pickle=True
        )

        return train_set, train_label, test_set, test_label


class RankDataHelper(DataHelper):
    def __init__(self, data_dir: str):
        super(RankDataHelper, self).__init__(data_dir)
        self.sub_dir = data_dir + "rank/"
        if not os.path.exists(self.sub_dir):
            os.makedirs(self.sub_dir)

    def gen_data_set(
        self,
        features: List[str],
        data: dict,
        test_id: dict,
        seq_max_len: int = 20,
        negnum: int = 0,
        min_rating: float = 0.35,
        n: int = 10,
    ):
        """Generate train set and test set

        Args:
            features (List[str]): feature list
            data (dict): data dictionary, keys: 'user', 'item', 'rating'
            test_id (dict): test id dictionary, {user_id: movie_id}. Note: each user should have the same number of movie_ids
            seq_max_len (int, optional): maximum history sequence length. Defaults to 20.
            negnum (int, optional): number of negative samples. Defaults to 0.
            min_rating (float, optional): minimum rating for positive smaples. Defaults to 0.35.
            n (int, optional): use the last n samples for each user to be the test set. Defaults to 10.

        """

        data["rating"].sort_values(by="timestamp", ascending=True, inplace=True)
        df = data["rating"]
        df["time_diff"] = df.groupby(["user_id"])["timestamp"].diff().fillna(0)

        # Split train set and test set
        item_ids = set(data["item"]["movie_id"].values)

        # Calculate number of rows of daraset to fasten the dataset generating process
        df = df[df["rating"] >= min_rating]
        counter = df[["user_id", "movie_id"]].groupby("user_id", as_index=False).count()
        counter = counter[counter["movie_id"] > n]
        df = df[df["user_id"].isin(counter["user_id"].values)]
        train_rows = ((counter["movie_id"] - n) * (negnum + 1)).sum()
        test_rows = len(test_id.keys()) * len(list(test_id.values())[0])

        # Generate rows
        # train_set format: [uid, moiveID, sample_age, time_since_last_movie, label, history_seq_len, history_seq]
        # test_set format: [uid, moiveID, sample_age(0), time_since_last_movie, history_seq_len, history_seq]
        train_set = np.zeros((train_rows, 6 + seq_max_len), dtype=int)
        test_set = np.zeros((test_rows, 5 + seq_max_len), dtype=int)

        p, q = 0, 0
        for uid, hist in tqdm(df.groupby("user_id"), "Generate train set"):
            pos_list = hist["movie_id"].tolist()
            time_diff_list = hist["time_diff"].tolist()
            if negnum > 0:
                candidate_set = list(item_ids - set(pos_list))  # Negative samples
                negs = np.random.choice(
                    candidate_set, size=len(pos_list) * negnum, replace=True
                )
            pos_list = pos_list[:-n]
            for i in range(len(pos_list)):
                seq = pos_list[:i]
                # Positive sample
                tmp_seq = seq[-seq_max_len:][::-1]
                train_set[p] = (
                    [
                        uid,
                        pos_list[i],
                        len(pos_list) - n - 1 - i,
                        time_diff_list[i],
                        1,
                        len(seq),
                    ]
                    + tmp_seq
                    + [0] * (seq_max_len - len(tmp_seq))
                )
                p += 1
                # Negative smaples
                for j in range(negnum):
                    train_set[p] = (
                        [
                            uid,
                            negs[j],
                            len(pos_list) - n - 1 - i,
                            time_diff_list[i],
                            0,
                            len(seq),
                        ]
                        + tmp_seq
                        + [0] * (seq_max_len - len(tmp_seq))
                    )
                    p += 1
            if uid in test_id.keys():
                for mid in test_id[uid]:
                    tmp_pos_list = pos_list[-seq_max_len:][::-1]
                    test_set[q] = (
                        [uid, mid, 0, time_diff_list[-1], len(pos_list)]
                        + tmp_pos_list
                        + [0] * (seq_max_len - len(tmp_pos_list))
                    )
                    q += 1

        np.random.seed(2022)
        np.random.shuffle(train_set)
        # np.random.shuffle(test_set)

        user = data["user"]
        item = data["item"]
        user = user.set_index("user_id")
        item = item.set_index("movie_id")

        train_uid = train_set[:, 0].astype(np.int)
        train_iid = train_set[:, 1].astype(np.int)
        train_age = train_set[:, 2].astype(np.int)
        train_time_gap = train_set[:, 3].astype(np.int)
        train_label = train_set[:, 4].astype(np.int)
        hist_seq_len = train_set[:, 5].astype(np.int)
        np.save(open(self.sub_dir + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.sub_dir + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.sub_dir + "train_example_age.npy", "wb"), train_age)
        np.save(open(self.sub_dir + "train_time_gap.npy", "wb"), train_time_gap)
        np.save(open(self.sub_dir + "train_label.npy", "wb"), train_label)
        np.save(open(self.sub_dir + "train_hist_movie_id_len.npy", "wb"), hist_seq_len)
        np.save(open(self.sub_dir + "train_hist_movie_id.npy", "wb"), train_set[:, 6:])

        test_uid = test_set[:, 0].astype(np.int)
        test_iid = test_set[:, 1].astype(np.int)
        test_age = test_set[:, 2].astype(np.int)
        test_time_gap = test_set[:, 3].astype(np.int)
        hist_seq_len = test_set[:, 4].astype(np.int)
        np.save(open(self.sub_dir + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.sub_dir + "test_movie_id.npy", "wb"), test_iid)
        np.save(open(self.sub_dir + "test_example_age.npy", "wb"), test_age)
        np.save(open(self.sub_dir + "test_time_gap.npy", "wb"), test_time_gap)
        np.save(open(self.sub_dir + "test_hist_movie_id_len.npy", "wb"), hist_seq_len)
        np.save(open(self.sub_dir + "test_hist_movie_id.npy", "wb"), test_set[:, 5:])

        del train_set, test_set  # , hist_seq, hist_seq_pad
        gc.collect()

        for key in tqdm([x for x in user.columns if x in features and x != "user_id"]):
            train_tmp_array = user[key].loc[train_uid].values
            test_tmp_array = user[key].loc[test_uid].values
            np.save(open(self.sub_dir + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.sub_dir + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

        del train_uid, user
        gc.collect()

        for key in tqdm([x for x in item.columns if x in features and x != "movie_id"]):
            train_tmp_array = item[key].loc[train_iid].values
            test_tmp_array = item[key].loc[test_iid].values
            np.save(open(self.sub_dir + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.sub_dir + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
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
            user_feats
            + ["hist_movie_id", "hist_movie_id_len", "time_gap", "example_age"],
            "Load user Features",
        ):
            train_set[feat] = np.load(
                open(self.sub_dir + "train_" + feat + ".npy", "rb"), allow_pickle=True
            )
            test_set[feat] = np.load(
                open(self.sub_dir + "test_" + feat + ".npy", "rb"), allow_pickle=True
            )

        for feat in tqdm(movie_feats, "Load movie Features"):
            train_set[feat] = np.load(
                open(self.sub_dir + "train_" + feat + ".npy", "rb"), allow_pickle=True
            )
            test_set[feat] = np.load(
                open(self.sub_dir + "test_" + feat + ".npy", "rb"), allow_pickle=True
            )

        train_label = np.load(
            open(self.sub_dir + "train_label.npy", "rb"), allow_pickle=True
        )

        return train_set, train_label, test_set


def sampledsoftmaxloss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


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
