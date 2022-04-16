from cgi import test
import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from tqdm import tqdm
from typing import Tuple, List, Dict
import numpy as np
import gc
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .datahelper import DataHelper


class MovielensDataHelper(DataHelper):
    """Base class for DataHelper for movielens dataset.

    Attributes
    ----------
    data_dir : str
        Diectory to load raw data and save generated dataset.
    """

    def __init__(self, data_dir: str):
        """Initialize a `MovielensDataHelper`

        Parameters
        ----------
        data_dir : str
            Diectory to load raw data and save generated dataset.
        """
        super().__init__(data_dir)

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

        return {"item": movies, "user": user, "inter": ratings}

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
        super().__init__(data_dir)

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
        data["inter"].sort_values("timestamp", inplace=True)
        df = data["inter"]
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
            negs = []
            if negnum > 0:
                candidate_set = list(item_ids - set(pos_list))  # * negative samples
                negs = np.random.choice(
                    candidate_set, size=(len(pos_list) - n) * negnum, replace=True
                )
            train_pos_list = pos_list[:-n]
            train_chunk = self._gen_smaple_chunk(uid, seq_max_len, train_pos_list, negs)
            train_set[p : p + train_chunk.shape[0]] = train_chunk
            p += train_chunk.shape[0]

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
        np.save(open(self.base + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.base + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.base + "train_example_age.npy", "wb"), train_age)
        np.save(open(self.base + "train_label.npy", "wb"), train_label)
        np.save(open(self.base + "train_hist_movie_id.npy", "wb"), train_set[:, 4:])

        test_uid = test_set[:, 0].astype(np.int)
        test_age = test_set[:, 1].astype(np.int)
        test_age = normalizer.transform(test_age.reshape(-1, 1))
        np.save(open(self.base + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.base + "test_example_age.npy", "wb"), test_age)
        np.save(open(self.base + "test_label.npy", "wb"), test_set[:, 2 : 2 + n])
        np.save(open(self.base + "test_hist_movie_id.npy", "wb"), test_set[:, 2 + n :])

        del train_set, test_set
        gc.collect()

        user_feats = [x for x in user.columns if x in features and x != "user_id"]
        for key in tqdm(user_feats, "Save user features"):
            train_tmp_array = np.array(user[key].loc[train_uid].tolist())
            test_tmp_array = np.array(user[key].loc[test_uid].tolist())
            np.save(open(self.base + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.base + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

        del train_uid, user
        gc.collect()

        item_feats = [x for x in item.columns if x in features and x != "movie_id"]
        for key in tqdm(item_feats, "Save item features"):
            train_tmp_array = np.array(item[key].loc[train_iid].tolist())
            np.save(open(self.base + "train_" + key + ".npy", "wb"), train_tmp_array)
            del train_tmp_array
            gc.collect()

    @staticmethod
    def _gen_smaple_chunk(uid, seq_len, pos_list, neg_list):
        """Generate a chunk of train set.

        Parameters
        ----------
        uid : int
            user_id
        seq_len : int
            movie history sequence length
        pos_list : List[int]
            positive movie ids
        neg_list : List[int]
            sampled negative movie ids

        Returns
        -------
        np.ndarray
            chunk of train set
        """

        def fill_n(x, i):
            top_n_seq = pos_list[:i][::-1][:seq_len]
            x[: len(top_n_seq)] = top_n_seq
            return x

        # * Trainset format: [uid, moiveID, sample_age, label, history_seq]
        # * Positive smaples
        len_pos = len(pos_list)
        chunk = np.zeros((len_pos, 4 + seq_len), dtype=int)
        chunk[:, 0] = uid  # * uid
        chunk[:, 1] = pos_list  # * movie_id
        chunk[:, 2] = len_pos - 1 - np.arange(len_pos)  # * sample age
        chunk[:, 3] = 1  # * label
        chunk[:, 4:] = np.array([fill_n(chunk[i, 4:], i) for i in range(len_pos)])

        # * Negative smaples
        if len(neg_list) > 0:
            neg_chunk = np.repeat(chunk, len(neg_list) // len_pos, axis=0)
            neg_chunk[:, 1] = neg_list  # * negative movie_id
            neg_chunk[:, 3] = 0  # * label
            chunk = np.concatenate((chunk, neg_chunk), axis=0)

        return chunk

    def load_dataset(self, user_feats: List[str], movie_feats: List[str]) -> Tuple:
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

        user_feats += ["hist_movie_id", "example_age"]
        for feat in tqdm(user_feats, "Load user features"):
            train_path = self.base + "train_" + feat + ".npy"
            test_path = self.base + "test_" + feat + ".npy"
            train_set[feat] = np.load(open(train_path, "rb"), allow_pickle=True)
            test_set[feat] = np.load(open(test_path, "rb"), allow_pickle=True)

        for feat in tqdm(movie_feats, "Load movie features"):
            train_path = self.base + "train_" + feat + ".npy"
            train_set[feat] = np.load(open(train_path, "rb"), allow_pickle=True)

        train_label_path = self.base + "train_label.npy"
        test_label_path = self.base + "test_label.npy"
        train_label = np.load(open(train_label_path, "rb"), allow_pickle=True)
        test_label = np.load(open(test_label_path, "rb"), allow_pickle=True)

        return train_set, train_label, test_set, test_label


class MovieRankDataHelper(MovielensDataHelper):
    def __init__(self, data_dir: str):
        super(MovieRankDataHelper, self).__init__(data_dir)

    def gen_dataset(
        self,
        features: List[str],
        data: dict,
        test_id: dict,
        seq_max_len: int = 20,
        negnum: int = 0,
        min_rating: float = 0.35,
        n: int = 10,
        neg_seq: bool = False,
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
        neg_seq : bool, optional
            Whether to generate negative sample sequences, by default ``False``.
            Set to ``True`` if you want to train `DIEN`.

        Note
        ----
        In ``test_id``, each user should have the same number of movie_ids.
        """

        data["inter"].sort_values(by="timestamp", ascending=True, inplace=True)
        df = data["inter"]
        df["time_diff"] = df.groupby(["user_id"])["timestamp"].diff().fillna(0)

        item_ids = set(data["item"]["movie_id"].values)

        # * Split train and test set
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

        train_dim = 5 + seq_max_len if not neg_seq else 5 + 2 * seq_max_len
        train_set = np.zeros((train_rows, train_dim), dtype=int)
        test_set = np.zeros((test_rows, train_dim - 1), dtype=int)

        p, q = 0, 0
        for uid, hist in tqdm(df.groupby("user_id"), "Generate train set"):
            pos_list = hist["movie_id"].tolist()
            time_diff = hist["time_diff"].tolist()
            negs = []
            if negnum > 0:
                candidate_set = list(item_ids - set(pos_list))  # * negative samples
                if not neg_seq:
                    neg_size = (len(pos_list) - n) * negnum
                else:
                    neg_size = (len(pos_list) - n) * (negnum + seq_max_len)
                negs = np.random.choice(candidate_set, size=neg_size, replace=True)

            pos_list = pos_list[:-n]
            time_diff = time_diff[:-n]
            test_list = test_id.get(uid, None)
            train_chunk = self._gen_train_chunk(
                uid, seq_max_len, pos_list, negs, time_diff, negnum, neg_seq
            )
            train_set[p : p + len(train_chunk)] = train_chunk
            p += len(train_chunk)
            if test_list is not None:  # * this uid needs to be tested
                test_chunk = self._gen_test_chunk(
                    uid, seq_max_len, pos_list, negs, test_list, time_diff, neg_seq
                )
                test_set[q : q + len(test_chunk)] = test_chunk
                q += len(test_chunk)

        np.random.seed(2022)
        np.random.shuffle(train_set)
        # np.random.shuffle(test_set)

        user = data["user"]
        item = data["item"]
        user = user.set_index("user_id")
        item = item.set_index("movie_id")

        age_normalizer = QuantileTransformer()
        gap_normalizer = QuantileTransformer()

        train_uid = train_set[:, 0]
        train_iid = train_set[:, 1]
        train_age = age_normalizer.fit_transform(train_set[:, 2].reshape(-1, 1))
        train_time_gap = gap_normalizer.fit_transform(train_set[:, 3].reshape(-1, 1))
        train_hist_seq = train_set[:, 5 : 5 + seq_max_len]
        np.save(open(self.base + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.base + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.base + "train_example_age.npy", "wb"), train_age)
        np.save(open(self.base + "train_time_gap.npy", "wb"), train_time_gap)
        np.save(open(self.base + "train_label.npy", "wb"), train_set[:, 4])
        np.save(open(self.base + "train_hist_movie_id.npy", "wb"), train_hist_seq)
        if neg_seq:
            train_neg_hist_seq = train_set[:, 5 + seq_max_len :]
            np.save(
                open(self.base + "train_neg_hist_movie_id.npy", "wb"),
                train_neg_hist_seq,
            )

        test_uid = test_set[:, 0]
        test_iid = test_set[:, 1]
        test_age = age_normalizer.transform(test_set[:, 2].reshape(-1, 1))
        test_time_gap = gap_normalizer.transform(test_set[:, 3].reshape(-1, 1))
        test_hist_seq = test_set[:, 4 : 4 + seq_max_len]
        np.save(open(self.base + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.base + "test_movie_id.npy", "wb"), test_iid)
        np.save(open(self.base + "test_example_age.npy", "wb"), test_age)
        np.save(open(self.base + "test_time_gap.npy", "wb"), test_time_gap)
        np.save(open(self.base + "test_hist_movie_id.npy", "wb"), test_hist_seq)
        if neg_seq:
            test_neg_hist_seq = test_set[:, 4 + seq_max_len :]
            np.save(
                open(self.base + "test_neg_hist_movie_id.npy", "wb"),
                test_neg_hist_seq,
            )

        del train_set, test_set  # , hist_seq, hist_seq_pad
        gc.collect()

        user_feats = [x for x in user.columns if x in features and x != "user_id"]
        for key in tqdm(user_feats, "Save user features"):
            train_tmp_array = np.array(user[key].loc[train_uid].tolist())
            test_tmp_array = np.array(user[key].loc[test_uid].tolist())
            np.save(open(self.base + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.base + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

        del train_uid, user
        gc.collect()

        item_feats = [x for x in item.columns if x in features and x != "movie_id"]
        for key in tqdm(item_feats, "Save item features"):
            train_tmp_array = np.array(item[key].loc[train_iid].tolist())
            test_tmp_array = np.array(item[key].loc[test_iid].tolist())
            np.save(open(self.base + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.base + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

    @staticmethod
    def _gen_train_chunk(
        uid, seq_len, pos_list, neg_list, time_diff_list, negnum, neg_seq
    ):
        """Generate train set of a single uid.

        Parameters
        ----------
        uid : int
            user_id
        seq_len : int
            movie history sequence length
        pos_list : List[int]
            positive movie ids
        neg_list : np.ndarray
            sampled negative movie ids
        time_diff_list : List[int]
            time gap since the last time to rate a movie
        neg_seq : bool
            whether to generate negative sample sequence

        Returns
        -------
        np.ndarray
            chunk of train set
        """

        def fill_n(x, i):
            top_n_seq = pos_list[:i][::-1][:seq_len]
            x[: len(top_n_seq)] = top_n_seq
            return x

        # * Trainset format: [uid, moiveID, sample_age, time_since_last_movie, label, history_seq]
        # * Positive smaples
        len_pos = len(pos_list)
        chunk_size = 5 + seq_len if not neg_seq else 5 + 2 * seq_len
        chunk = np.zeros((len_pos, chunk_size), dtype=int)
        chunk[:, 0] = uid  # * uid
        chunk[:, 1] = pos_list  # * movieID
        chunk[:, 2] = len_pos - 1 - np.arange(len_pos)  # * sample_age
        chunk[:, 2] = len_pos - 1 - np.arange(len_pos)  # * sample age
        chunk[:, 3] = time_diff_list  # * time since last watched movie
        chunk[:, 4] = 1  # * label
        chunk[:, 5 : 5 + seq_len] = np.array(
            [fill_n(chunk[i, 5 : 5 + seq_len], i) for i in range(len_pos)]
        )
        if neg_seq:
            chunk[:, 5 + seq_len :] = neg_list[: len_pos * seq_len].reshape(len_pos, -1)

        # * Negative smaples
        if len(neg_list) > 0:
            neg_chunk = np.repeat(chunk, negnum, axis=0)
            neg_chunk[:, 1] = neg_list[-negnum * len_pos :]  # * negative movie_id
            neg_chunk[:, 4] = 0  # * label
            chunk = np.concatenate((chunk, neg_chunk), axis=0)

        return chunk

    @staticmethod
    def _gen_test_chunk(uid, seq_len, pos_list, negs, test_list, time_diff, neg_seq):
        """Generate test set of a single uid.

        Parameters
        ----------
        uid : int
            user_id
        seq_len : int
            movie history sequence length
        pos_list : List[int]
            positive movie ids
        test_list : List[int]
            test movie ids
        time_diff : List[int]
            time gap since the last time to rate a movie

        Returns
        -------
        np.ndarray
            chunk of test set
        """
        # * Testset format: [uid, moiveID, sample_age(0), time_since_last_movie, history_seq]
        # * Test smaples
        hist_seq = np.array(pos_list[-seq_len:][::-1])
        hist_seq_len = len(hist_seq)
        test_len = len(test_list)
        chunk_size = 4 + seq_len if not neg_seq else 4 + 2 * seq_len
        test_chunk = np.zeros((test_len, chunk_size), dtype=int)
        test_chunk[:, 0] = uid  # * uid
        test_chunk[:, 1] = test_list  # * movie_id
        test_chunk[:, 3] = time_diff[-1]  # * time since last watched movie
        test_chunk[:, 4 : 4 + hist_seq_len] = hist_seq  # * history_seq
        if neg_seq:
            neg_hist_seq = negs[:seq_len]
            test_chunk[:, 4 + seq_len :] = neg_hist_seq

        return test_chunk

    def load_dataset(
        self,
        user_feats: List[str],
        movie_feats: List[str],
        neg_seq: bool = False,
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

        u_feats = user_feats.copy()
        if neg_seq:
            u_feats += ["neg_hist_movie_id"]
        u_feats += ["hist_movie_id", "time_gap", "example_age"]
        for feat in tqdm(u_feats, "Load user features"):
            train_path = self.base + "train_" + feat + ".npy"
            test_path = self.base + "test_" + feat + ".npy"
            train_set[feat] = np.load(open(train_path, "rb"), allow_pickle=True)
            test_set[feat] = np.load(open(test_path, "rb"), allow_pickle=True)

        for feat in tqdm(movie_feats, "Load movie features"):
            train_path = self.base + "train_" + feat + ".npy"
            test_path = self.base + "test_" + feat + ".npy"
            train_set[feat] = np.load(open(train_path, "rb"), allow_pickle=True)
            test_set[feat] = np.load(open(test_path, "rb"), allow_pickle=True)

        train_label_path = self.base + "train_label.npy"
        train_label = np.load(open(train_label_path, "rb"), allow_pickle=True)

        return train_set, train_label, test_set


class MovieRankSeqDataHelper(MovielensDataHelper):
    def __init__(self, data_dir: str):
        super().__init__(data_dir, "rank")

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

        Note
        ----
        In ``test_id``, each user should have the same number of movie_ids.
        """

        data["inter"].sort_values(by="timestamp", ascending=True, inplace=True)
        df = data["inter"]

        item_ids = set(data["item"]["movie_id"].values)

        # * Split train and test set
        # * Calculate number of rows of dataset to fasten the dataset generating process
        df = df[df["interact"] >= min_rating]
        counter = df[["user_id", "movie_id"]].groupby("user_id", as_index=False).count()
        counter = counter[counter["movie_id"] > n]
        df = df[df["user_id"].isin(counter["user_id"].values)]
        train_rows = ((counter["movie_id"] - n) * negnum).sum()
        test_rows = len(test_id.keys()) * len(list(test_id.values())[0])

        # * Generate rows
        # * train_set format: [uid, movie_id, neg_movie_id, history_seq]
        # * test_set format: [uid, moiveID, history_seq]

        train_set = np.zeros((train_rows, 3 + seq_max_len), dtype=int)
        test_set = np.zeros((test_rows, 2 + seq_max_len), dtype=int)

        p, q = 0, 0
        for uid, hist in tqdm(df.groupby("user_id"), "Generate train set"):
            pos_list = hist["movie_id"].tolist()
            negs = []
            if negnum > 0:
                candidate_set = list(item_ids - set(pos_list))  # * negative samples
                neg_size = (len(pos_list) - n) * negnum
                negs = np.random.choice(candidate_set, size=neg_size, replace=True)

            pos_list = pos_list[:-n]
            test_list = test_id.get(uid, None)
            train_chunk = self._gen_train_chunk(
                uid, seq_max_len, pos_list, negs, negnum
            )
            train_set[p : p + len(train_chunk), :] = train_chunk
            p += len(train_chunk)
            if test_list is not None:  # * this uid needs to be tested
                test_chunk = self._gen_test_chunk(uid, seq_max_len, pos_list, test_list)
                test_set[q : q + len(test_chunk), :] = test_chunk
                q += len(test_chunk)

        np.random.seed(2022)
        np.random.shuffle(train_set)
        # np.random.shuffle(test_set)

        user = data["user"]
        item = data["item"]
        user = user.set_index("user_id")
        item = item.set_index("movie_id")

        train_uid = train_set[:, 0]
        train_iid = train_set[:, 1]
        train_hist_seq = train_set[:, 3 : 3 + seq_max_len]
        np.save(open(self.base + "train_user_id.npy", "wb"), train_uid)
        np.save(open(self.base + "train_movie_id.npy", "wb"), train_iid)
        np.save(open(self.base + "train_neg_movie_id.npy", "wb"), train_set[:, 2])
        np.save(open(self.base + "train_hist_movie_id.npy", "wb"), train_hist_seq)

        test_uid = test_set[:, 0]
        test_iid = test_set[:, 1]
        test_hist_seq = test_set[:, 2 : 2 + seq_max_len]
        np.save(open(self.base + "test_user_id.npy", "wb"), test_uid)
        np.save(open(self.base + "test_movie_id.npy", "wb"), test_iid)
        np.save(open(self.base + "test_hist_movie_id.npy", "wb"), test_hist_seq)

        del train_set, test_set  # , hist_seq, hist_seq_pad
        gc.collect()

        user_feats = [x for x in user.columns if x in features and x != "user_id"]
        for key in tqdm(user_feats, "Save user features"):
            train_tmp_array = np.array(user[key].loc[train_uid].tolist())
            test_tmp_array = np.array(user[key].loc[test_uid].tolist())
            np.save(open(self.base + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.base + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

        del train_uid, user
        gc.collect()

        item_feats = [x for x in item.columns if x in features and x != "movie_id"]
        for key in tqdm(item_feats, "Save item features"):
            train_tmp_array = np.array(item[key].loc[train_iid].tolist())
            test_tmp_array = np.array(item[key].loc[test_iid].tolist())
            np.save(open(self.base + "train_" + key + ".npy", "wb"), train_tmp_array)
            np.save(open(self.base + "test_" + key + ".npy", "wb"), test_tmp_array)
            del train_tmp_array, test_tmp_array
            gc.collect()

    @staticmethod
    def _gen_train_chunk(uid, seq_len, pos_list, neg_list, negnum):
        """Generate train set of a single uid.

        Parameters
        ----------
        uid : int
            user_id
        seq_len : int
            movie history sequence length
        pos_list : List[int]
            positive movie ids
        neg_list : np.ndarray
            sampled negative movie ids

        Returns
        -------
        np.ndarray
            chunk of train set
        """

        def fill_n(x, i):
            top_n_seq = pos_list[:i][::-1][:seq_len]
            x[: len(top_n_seq)] = top_n_seq
            return x

        # * Trainset format: [uid, moive_id, neg_movie_id, history_seq]
        # * Positive smaples
        len_pos = len(pos_list)
        chunk_size = 3 + seq_len
        chunk = np.zeros((len_pos * negnum, chunk_size), dtype=int)
        chunk[:, 0] = uid  # * uid
        chunk[:, 1] = np.repeat(pos_list, negnum, -1)  # * movieID
        chunk[:, 2] = neg_list[-negnum * len_pos :]  # * negative movieID
        chunk[:, 3 : 3 + seq_len] = np.repeat(
            np.array([fill_n(chunk[i, 3 : 3 + seq_len], i) for i in range(len_pos)]),
            negnum,
            0,
        )

        return chunk

    @staticmethod
    def _gen_test_chunk(uid, seq_len, pos_list, test_list):
        """Generate test set of a single uid.

        Parameters
        ----------
        uid : int
            user_id
        seq_len : int
            movie history sequence length
        pos_list : List[int]
            positive movie ids
        test_list : List[int]
            test movie ids
        time_diff : List[int]
            time gap since the last time to rate a movie

        Returns
        -------
        np.ndarray
            chunk of test set
        """
        # * Testset format: [uid, moiveID, sample_age(0), time_since_last_movie, history_seq]
        # * Test smaples
        hist_seq = np.array(pos_list[-seq_len:][::-1])
        hist_seq_len = len(hist_seq)
        test_len = len(test_list)
        chunk_size = 2 + seq_len
        test_chunk = np.zeros((test_len, chunk_size), dtype=int)
        test_chunk[:, 0] = uid  # * uid
        test_chunk[:, 1] = test_list  # * movie_id
        test_chunk[:, 2 : 2 + hist_seq_len] = hist_seq  # * history_seq

        return test_chunk

    def load_dataset(self, user_feats: List[str], movie_feats: List[str]) -> Tuple:
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

        user_feats += ["hist_movie_id"]
        for feat in tqdm(user_feats, "Load user features"):
            train_path = self.base + "train_" + feat + ".npy"
            test_path = self.base + "test_" + feat + ".npy"
            train_set[feat] = np.load(open(train_path, "rb"), allow_pickle=True)
            test_set[feat] = np.load(open(test_path, "rb"), allow_pickle=True)

        for feat in tqdm(movie_feats, "Load movie features"):
            train_path = self.base + "train_" + feat + ".npy"
            test_path = self.base + "test_" + feat + ".npy"
            train_set[feat] = np.load(open(train_path, "rb"), allow_pickle=True)
            test_set[feat] = np.load(open(test_path, "rb"), allow_pickle=True)
        train_set["neg_movie_id"] = np.load(
            open(self.base + "train_neg_movie_id.npy", "rb"), allow_pickle=True
        )

        return train_set, test_set
