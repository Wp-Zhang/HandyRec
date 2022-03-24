import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from tqdm import tqdm
from typing import Tuple, List
import numpy as np
import gc
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from handyrec.dataset import DataHelper
from typing import Dict


class MovielensDataHelper(DataHelper):
    """base class for DataHelper for movielens dataset"""

    def __init__(self, data_dir: str, sub_dir_name: str):
        super(MovielensDataHelper, self).__init__(data_dir)
        self.sub_dir = data_dir + sub_dir_name + "/"
        if not os.path.exists(self.sub_dir):
            os.makedirs(self.sub_dir)

    def load_data(self) -> Dict:
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
        genres = movies["genres"].str.get_dummies(sep="|")
        movies = pd.concat([movies[["movie_id", "title"]], genres], axis=1)

        return {"item": movies, "user": user, "interact": ratings}

    def preprocess_data(self, data: dict, sparse_features: List[str]) -> dict:
        """Preprocess raw data

        Args:
            data (dict): data dictionary, keys: 'item', 'user', 'interact'
            sparse_features (List[str]): sparse feature list to be label encoded

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

    def get_clean_data(self, sparse_features: List[str]) -> Dict:
        """Load raw data and preprocess

        Args:
            sparse_features (List[str]): sparse feature list to be label encoded

        Returns:
            Dict: a dictionary of preprocessed data with three keys: [`user`, `item`, `interact`]
        """
        data = self.load_data()
        data = self.preprocess_data(data, sparse_features)
        return data


class MovieMatchDataHelper(MovielensDataHelper):
    def __init__(self, data_dir: str):
        super(MovieMatchDataHelper, self).__init__(data_dir, "match")

    def gen_dataset(
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
            data (dict): data dictionary, keys: 'user', 'item', 'interact'
            seq_max_len (int, optional): maximum history sequence length. Defaults to 20.
            min_rating (float, optional): minimum interact for positive smaples. Defaults to 0.35.
            n (int, optional): use the last n samples for each user to be the test set. Defaults to 10.

        """

        data["interact"].sort_values("timestamp", inplace=True)
        df = data["interact"]

        # * Calculate number of rows of dataset to fasten the dataset generating process
        df = df[df["interact"] >= min_rating]
        counter = df[["user_id", "movie_id"]].groupby("user_id", as_index=False).count()
        counter = counter[counter["movie_id"] > n]
        df = df[df["user_id"].isin(counter["user_id"].values)]
        train_rows = (counter["movie_id"] - n).sum()
        test_rows = counter.shape[0]

        # * Generate rows
        # * train_set format: [uid, moiveID, sample_age, label, history_seq]
        # * test_set format: [uid, moiveIDs, sample_age(0), history_seq]
        train_set = np.zeros((train_rows, 5), dtype=object)
        test_set = np.zeros((test_rows, 4), dtype=object)

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
                    seq[::-1],
                ]
                p += 1
            i = len(pos_list) - n
            test_set[q] = [uid, pos_list[i:], 0, pos_list[:i][::-1]]
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
        hist_seq = train_set[:, 4].tolist()
        hist_seq_pad = pad_sequences(
            hist_seq, maxlen=seq_max_len, padding="post", truncating="post", value=0
        )
        normalizer = QuantileTransformer()
        train_age = normalizer.fit_transform(train_age.reshape(-1, 1))
        np.save(open(self.sub_dir + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.sub_dir + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.sub_dir + "train_example_age.npy", "wb"), train_age)
        np.save(open(self.sub_dir + "train_label.npy", "wb"), train_label)
        np.save(open(self.sub_dir + "train_hist_movie_id.npy", "wb"), hist_seq_pad)

        test_uid = test_set[:, 0].astype(np.int)
        test_label = np.array(test_set[:, 1].tolist()).astype(np.int)
        test_age = test_set[:, 2].astype(np.int)
        hist_seq = test_set[:, 3].tolist()
        hist_seq_pad = pad_sequences(
            hist_seq, maxlen=seq_max_len, padding="post", truncating="post", value=0
        )
        test_age = normalizer.transform(test_age.reshape(-1, 1))
        np.save(open(self.sub_dir + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.sub_dir + "test_example_age.npy", "wb"), test_age)
        np.save(open(self.sub_dir + "test_label.npy", "wb"), test_label)
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
        """Generate train set and test set

        Args:
            features (List[str]): feature list
            data (dict): data dictionary, keys: 'user', 'item', 'interact'
            test_id (dict): test id dictionary, {user_id: movie_id}. Note: each user should have the same number of movie_ids
            seq_max_len (int, optional): maximum history sequence length. Defaults to 20.
            negnum (int, optional): number of negative samples. Defaults to 0.
            min_rating (float, optional): minimum interact for positive smaples. Defaults to 0.35.
            n (int, optional): use the last n samples for each user to be the test set. Defaults to 10.

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
