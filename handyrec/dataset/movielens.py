import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from tqdm import tqdm
from typing import Tuple, List, Dict
import numpy as np
import gc
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .datahelper import DataHelper


class MovielensDataHelper(DataHelper):
    """Base class for DataHelper for movielens dataset.

    Attributes
    ----------
    data_dir : str
        Diectory to load raw data and save generated dataset.
    sub_dir_name : str
        Diectory to save and load generated feature values for training.
    """

    def __init__(self, data_dir: str, sub_dir_name: str):
        """Initialize a `MovielensDataHelper`

        Parameters
        ----------
        data_dir : str
            Diectory to load raw data and save generated dataset.
        sub_dir_name : str
            Diectory to save and load generated feature values for training.
        """
        super().__init__(data_dir)
        self.sub_dir = data_dir + sub_dir_name + "/"
        if not os.path.exists(self.sub_dir):
            os.makedirs(self.sub_dir)

    def load_data(self) -> Dict:
        """Load original raw data.

        Returns
        -------
        Dict
            Dictionary of raw data with three keys: ``user``, ``item``, and ``interact``.
        """
        unames = ["user_id", "gender", "age", "occupation", "zip"]
        user = pd.read_csv(
            self.base + "users.dat",
            sep="::",
            header=None,
            names=unames,
            encoding="latin-1",
            engine="python",
        )
        rnames = ["user_id", "movie_id", "interact", "timestamp"]
        ratings = pd.read_csv(
            self.base + "ratings.dat",
            sep="::",
            header=None,
            names=rnames,
            encoding="latin-1",
            engine="python",
        )
        mnames = ["movie_id", "title", "genres"]
        movies = pd.read_csv(
            self.base + "movies.dat",
            sep="::",
            header=None,
            names=mnames,
            encoding="latin-1",
            engine="python",
        )
        movies["year"] = movies["title"].str.slice(-5, -1).astype(int)
        genres = list(movies["genres"].str.get_dummies(sep="|").columns)
        genre_map = {x: i + 1 for i, x in enumerate(genres)}  # index 0 is for padding
        movies["genres"] = movies["genres"].apply(
            lambda x: sorted([genre_map[k] for k in x.split("|")])
        )
        pad_genres = pad_sequences(movies["genres"], padding="post")
        movies["genres"] = (
            movies.reset_index()
            .pop("index")
            .apply(lambda x: pad_genres[x - 1].tolist())
        )

        return {"item": movies, "user": user, "interact": ratings}

    def preprocess_data(self, data: dict, sparse_features: List[str]) -> Dict:
        """Preprocess raw data

        Parameters
        ----------
        data : dict
            Dictionary with three keys: ``user``, ``item``, and ``interact``.
        sparse_features : List[str]
            List of sparse features to be label encoded.

        Returns
        -------
        Dict
            Dictionary of processed data with three keys: ``user``, ``item``, and ``interact``.
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

    def get_clean_data(self, sparse_features: List[str]) -> Dict:
        """Wrapper for load and preprocess data.

        Parameters
        ----------
        sparse_features : List[str]
            List of sparse features to be label encoded.

        Returns
        -------
        Dict
            Dictionary of processed data with three keys: ``user``, ``item``, and ``interact``.
        """
        data = self.load_data()
        data = self.preprocess_data(data, sparse_features)
        return data


class MovieMatchDataHelper(MovielensDataHelper):
    """DataHelper for generating movielens dataset for matching models."""

    def __init__(self, data_dir: str):
        """Initialize a ``MovieMatchDataHelper``.

        Parameters
        ----------
        data_dir : str
            Diectory to load raw data and save generated dataset.
        """
        super().__init__(data_dir, "match")

    def gen_dataset(
        self,
        features: List[str],
        data: dict,
        seq_max_len: int = 20,
        negnum: int = 0,
        min_rating: float = 0.35,
        n: int = 10,
    ):
        """Generate and save train set and test set.

        Parameters
        ----------
        features : List[str]
            List of features to be contained in the dataset.
        data : dict
            Data dictionary with three keys: ``user``, ``item``, and ``interact``.
        seq_max_len : int, optional
            Maximum history sequence length, by default ``20``.
        negnum : int, optional
            Number of negative samples, by default ``0``.
        min_rating : float, optional
            Minimum rating for positive smaples, by default ``0.35``.
        n : int, optional
            Hold out the last n samples for each user for testing, by default ``10``.
        """
        data["interact"].sort_values("timestamp", inplace=True)
        df = data["interact"]
        item_ids = set(data["item"]["movie_id"].values)

        # * Calculate number of rows of dataset to fasten the dataset generating process
        df = df[df["interact"] >= min_rating]
        counter = df[["user_id", "movie_id"]].groupby("user_id", as_index=False).count()
        counter = counter[counter["movie_id"] > n]
        df = df[df["user_id"].isin(counter["user_id"].values)]
        train_rows = ((counter["movie_id"] - n) * (negnum + 1)).sum()
        test_rows = counter.shape[0]

        # * Generate rows
        # * train_set format: [uid, moiveID, sample_age, label, history_seq]
        # * test_set format: [uid, sample_age(0), moiveIDs, history_seq]
        train_set = np.zeros((train_rows, 4 + seq_max_len), dtype=int)
        test_set = np.zeros((test_rows, 2 + n + seq_max_len), dtype=int)

        p, q = 0, 0
        for uid, hist in tqdm(df.groupby("user_id"), "Generate train set"):
            pos_list = hist["movie_id"].tolist()
            if negnum > 0:
                candidate_set = list(item_ids - set(pos_list))  # Negative samples
                negs = np.random.choice(
                    candidate_set, size=(len(pos_list) - n) * negnum, replace=True
                )
            train_pos_list = pos_list[:-n]
            for i in range(len(train_pos_list)):
                seq = train_pos_list[:i]
                # Positive sample
                tmp_seq = seq[-seq_max_len:][::-1]
                train_set[p] = (
                    [
                        uid,
                        train_pos_list[i],
                        len(train_pos_list) - 1 - i,
                        1,
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
                            negs[i * negnum + j],
                            len(pos_list) - 1 - i,
                            0,
                        ]
                        + tmp_seq
                        + [0] * (seq_max_len - len(tmp_seq))
                    )
                    p += 1
            test_pos_list = pos_list[-seq_max_len - n : -n][::-1]
            test_pos_list += [0] * (seq_max_len - len(test_pos_list))
            test_set[q] = [uid, 0] + pos_list[-n:] + test_pos_list
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
        normalizer = QuantileTransformer()
        train_age = normalizer.fit_transform(train_age.reshape(-1, 1))
        np.save(open(self.sub_dir + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.sub_dir + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.sub_dir + "train_example_age.npy", "wb"), train_age)
        np.save(open(self.sub_dir + "train_label.npy", "wb"), train_label)
        np.save(open(self.sub_dir + "train_hist_movie_id.npy", "wb"), train_set[:, 4:])

        test_uid = test_set[:, 0].astype(np.int)
        test_age = test_set[:, 1].astype(np.int)
        test_age = normalizer.transform(test_age.reshape(-1, 1))
        np.save(open(self.sub_dir + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.sub_dir + "test_example_age.npy", "wb"), test_age)
        np.save(open(self.sub_dir + "test_label.npy", "wb"), test_set[:, 2 : 2 + n])
        np.save(
            open(self.sub_dir + "test_hist_movie_id.npy", "wb"), test_set[:, 2 + n :]
        )

        del train_set, test_set
        gc.collect()

        for key in tqdm([x for x in user.columns if x in features and x != "user_id"]):
            train_tmp_array = np.array(user[key].loc[train_uid].tolist())
            test_tmp_array = np.array(user[key].loc[test_uid].tolist())
            np.save(open(self.sub_dir + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.sub_dir + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

        del train_uid, user
        gc.collect()

        for key in tqdm([x for x in item.columns if x in features and x != "movie_id"]):
            train_tmp_array = np.array(item[key].loc[train_iid].tolist())
            np.save(open(self.sub_dir + "train_" + key + ".npy", "wb"), train_tmp_array)
            del train_tmp_array
            gc.collect()

    def load_dataset(
        self,
        user_feats: List[str],
        movie_feats: List[str],
    ) -> Tuple:
        """Load saved dataset.

        Parameters
        ----------
        user_feats : List[str]
            List of user features to be loaded.
        movie_feats : List[str]
            List of movie features to be loaded.

        Returns
        -------
        Tuple
            [train set, test set].
        """

        train_set = {}
        test_set = {}

        for feat in tqdm(
            user_feats + ["hist_movie_id", "example_age"],
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


class MovieRankDataHelper(MovielensDataHelper):
    def __init__(self, data_dir: str):
        super(MovieRankDataHelper, self).__init__(data_dir, "rank")

    def gen_dataset(
        self,
        features: List[str],
        data: dict,
        test_id: dict,
        seq_max_len: int = 20,
        negnum: int = 0,
        min_rating: float = 0.35,
        n: int = 10,
    ):
        """Generate and save train set and test set.

        Parameters
        ----------
        features : List[str]
            List of features to be contained in the dataset.
        data : dict
            Data dictionary with three keys: ``user``, ``item``, and ``interact``.
        test_id : dict
            Test id dictionary, {user_id: movie_id}.
        seq_max_len : int, optional
            Maximum history sequence length, by default ``20``.
        negnum : int, optional
            Number of negative samples, by default ``0``.
        min_rating : float, optional
            Minimum rating for positive smaples, by default ``0.35``.
        n : int, optional
            Hold out the last n samples for each user for testing, by default ``10``.

        Notes
        -----
        In ``test_id``, each user should have the same number of movie_ids.
        """

        data["interact"].sort_values(by="timestamp", ascending=True, inplace=True)
        df = data["interact"]
        df["time_diff"] = df.groupby(["user_id"])["timestamp"].diff().fillna(0)

        # * Split train set and test set
        item_ids = set(data["item"]["movie_id"].values)

        # * Calculate number of rows of dataset to fasten the dataset generating process
        df = df[df["interact"] >= min_rating]
        counter = df[["user_id", "movie_id"]].groupby("user_id", as_index=False).count()
        counter = counter[counter["movie_id"] > n]
        df = df[df["user_id"].isin(counter["user_id"].values)]
        train_rows = ((counter["movie_id"] - n) * (negnum + 1)).sum()
        test_rows = len(test_id.keys()) * len(list(test_id.values())[0])

        # * Generate rows
        # * train_set format: [uid, moiveID, sample_age, time_since_last_movie, label, history_seq]
        # * test_set format: [uid, moiveID, sample_age(0), time_since_last_movie, history_seq]
        train_set = np.zeros((train_rows, 5 + seq_max_len), dtype=int)
        test_set = np.zeros((test_rows, 4 + seq_max_len), dtype=int)

        p, q = 0, 0
        for uid, hist in tqdm(df.groupby("user_id"), "Generate train set"):
            pos_list = hist["movie_id"].tolist()
            time_diff_list = hist["time_diff"].tolist()
            if negnum > 0:
                candidate_set = list(item_ids - set(pos_list))  # Negative samples
                negs = np.random.choice(
                    candidate_set, size=(len(pos_list) - n) * negnum, replace=True
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
                        len(pos_list) - 1 - i,
                        time_diff_list[i],
                        1,
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
                            negs[i * negnum + j],
                            len(pos_list) - 1 - i,
                            time_diff_list[i],
                            0,
                        ]
                        + tmp_seq
                        + [0] * (seq_max_len - len(tmp_seq))
                    )
                    p += 1
            if uid in test_id.keys():
                for mid in test_id[uid]:
                    tmp_pos_list = pos_list[-seq_max_len:][::-1]
                    test_set[q] = (
                        [uid, mid, 0, time_diff_list[-1]]
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
        normalizer = QuantileTransformer()
        normalizer2 = QuantileTransformer()
        train_age = normalizer.fit_transform(train_age.reshape(-1, 1))
        train_time_gap = normalizer2.fit_transform(train_time_gap.reshape(-1, 1))
        np.save(open(self.sub_dir + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.sub_dir + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.sub_dir + "train_example_age.npy", "wb"), train_age)
        np.save(open(self.sub_dir + "train_time_gap.npy", "wb"), train_time_gap)
        np.save(open(self.sub_dir + "train_label.npy", "wb"), train_label)
        np.save(open(self.sub_dir + "train_hist_movie_id.npy", "wb"), train_set[:, 5:])

        test_uid = test_set[:, 0].astype(np.int)
        test_iid = test_set[:, 1].astype(np.int)
        test_age = test_set[:, 2].astype(np.int)
        test_time_gap = test_set[:, 3].astype(np.int)
        test_age = normalizer.transform(test_age.reshape(-1, 1))
        test_time_gap = normalizer2.transform(test_time_gap.reshape(-1, 1))
        np.save(open(self.sub_dir + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.sub_dir + "test_movie_id.npy", "wb"), test_iid)
        np.save(open(self.sub_dir + "test_example_age.npy", "wb"), test_age)
        np.save(open(self.sub_dir + "test_time_gap.npy", "wb"), test_time_gap)
        np.save(open(self.sub_dir + "test_hist_movie_id.npy", "wb"), test_set[:, 4:])

        del train_set, test_set  # , hist_seq, hist_seq_pad
        gc.collect()

        for key in tqdm([x for x in user.columns if x in features and x != "user_id"]):
            train_tmp_array = np.array(user[key].loc[train_uid].tolist())
            test_tmp_array = np.array(user[key].loc[test_uid].tolist())
            np.save(open(self.sub_dir + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.sub_dir + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

        del train_uid, user
        gc.collect()

        for key in tqdm([x for x in item.columns if x in features and x != "movie_id"]):
            train_tmp_array = np.array(item[key].loc[train_iid].tolist())
            test_tmp_array = np.array(item[key].loc[test_iid].tolist())
            np.save(open(self.sub_dir + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.sub_dir + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

    def load_dataset(
        self,
        user_feats: List[str],
        movie_feats: List[str],
    ) -> Tuple:
        """Load saved dataset.

        Parameters
        ----------
        user_feats : List[str]
            List of user features to be loaded.
        movie_feats : List[str]
            List of movie features to be loaded.

        Returns
        -------
        Tuple
            [train set, test set].
        """
        train_set = {}
        test_set = {}

        for feat in tqdm(
            user_feats + ["hist_movie_id", "time_gap", "example_age"],
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
