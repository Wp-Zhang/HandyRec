from typing import List, Dict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
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
