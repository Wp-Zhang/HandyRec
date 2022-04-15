from typing import Dict, List, Tuple, Any
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import gc
from copy import deepcopy


class HandyRecDataset:
    """The base class for all types of datasets.

    The order of calling the methods should be:
        (0. `negative_sampling`)

        1. `train_test_split`

        2. `train_valid_split`

        3. `gen_dataset`

        4. `load_dataset`

    Attibutes
    ---------
    dir: Path
        The directory of the dataset.
    data: Dict
        The data of the dataset.
    uid_name: str
        The name of the user id column.
    iid_name: str
        The name of the item id column.
    inter_name: str
        The name of the interaction column.
    time_name: str
        The name of the time column.
    threshold: float
        The threshold for the interactions.
    """

    def __init__(
        self,
        name: str,
        task: str,
        data: Dict,
        uid_name: str,
        iid_name: str,
        inter_name: str,
        time_name: str,
        threshold: float = 0.0,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        name: str
            The name of the dataset.
        task: str
            The task of the dataset, should be one of {``retrieval``, ``ranking``}.
        data: Dict
            The data of the dataset.
        uid_name: str
            The name of the user id column.
        iid_name: str
            The name of the item id column.
        inter_name: str
            The name of the interaction column.
        time_name: str
            The name of the time column.
        threshold: float, optional
            The threshold for the interactions, by default ``0.0``.
        """
        self._check_task(task)
        self._check_data(data)
        self._check_path(name)

        self.task = task
        self.dir = Path(name)
        self.data = deepcopy(data)
        self.uid_name = uid_name
        self.iid_name = iid_name
        self.inter_name = inter_name
        self.time_name = time_name
        self.threshold = threshold

        mask = self.data["inter"][inter_name] >= threshold
        self.data["inter"] = (
            self.data["inter"]
            .loc[mask]
            .sort_values(by=[uid_name, time_name, iid_name])
            .reset_index(drop=True)
        )

        self.index = {"train": [], "valid": []}
        self.test_inter = pd.DataFrame()
        self.test_label = None

    # * ==================== Check validity of the parameters =====================
    def _check_task(self, task) -> None:
        """Check if the task is valid.

        Parameters
        ----------
        task: str
            The task of the dataset, should be one of {``retrieval``, ``ranking``}.
        """
        if task not in ["retrieval", "ranking"]:
            raise ValueError("The task should be one of {``retrieval``, ``ranking``}")

    def _check_data(self, data: Dict) -> None:
        """Check if the data is valid.

        Parameters
        ----------
        data: Dict
            The data of the dataset.

        Raises
        ------
        KeyError
            If one of keys in {``user``, ``item``, ``inter``} is not in `data`.
        """
        keys = set(data.keys())
        res = set(["user", "item", "inter"]) - keys
        if len(res) > 0:
            raise KeyError("The following keys are missing: {}".format(", ".join(res)))
        if len(keys) > 3:
            warnings.warn(
                "The following keys will not be used: {}".format(", ".join(keys - res))
            )

    def _check_path(self, name: str) -> None:
        """Check if the path is valid.

        Parameters
        ----------
        path: str
            The path to be checked.
        """
        if Path(name).exists():
            warnings.warn(
                "The dataset {} already exists, will be overwritten".format(name)
            )
        else:
            Path(name).mkdir()

    def _check_features(self, data_name: str, features: List[str]) -> None:
        """Check if the features are valid.

        Parameters
        ----------
        data_name: str
            The name of the data, should be one of {``user``, ``item``, ``inter``}.
        features: List[str]
            The features to be checked.

        Raises
        ------
        KeyError
            If some of the features is not in the data.
        """
        for name in features:
            if name not in self.data[data_name].keys():
                raise KeyError(
                    "The feature {} is not in the {} dataset".format(name, data_name)
                )

    # * ========================================================================== *

    def _save_features(self, name: str, features: List[str]) -> Tuple[Dict]:
        """Save the features of the dataset.

        Parameters
        ----------
        name: str
            The name of the dataset, should be one of {``user``, ``item``, ``inter``}.
        features: List[str]
            The features to be saved.

        Returns
        -------
        Tuple[Dict]
            The features of the dataset. (train, valid, test)
        """
        if name == "user":
            data = self.data[name].set_index(self.uid_name)
            data[self.uid_name] = data.index
        elif name == "item":
            data = self.data[name].set_index(self.iid_name)
            data[self.iid_name] = data.index
        else:
            data = self.data[name]

        train_dict = {}
        valid_dict = {}
        test_dict = {}

        for feat in tqdm(features, f"Save {name} features"):
            train_idx = self.index["train"]
            valid_idx = self.index["valid"]

            if name == "user":
                train_idx = self.data["inter"][self.uid_name].loc[train_idx].values
                valid_idx = self.data["inter"][self.uid_name].loc[valid_idx].values
                test_idx = self.test_inter[self.uid_name].values
            elif name == "item":
                train_idx = self.data["inter"][self.iid_name].loc[train_idx].values
                valid_idx = self.data["inter"][self.iid_name].loc[valid_idx].values
                test_idx = self.test_inter[self.iid_name].values

            train_tmp_array = np.array(data[feat].loc[train_idx].tolist())
            valid_tmp_array = np.array(data[feat].loc[valid_idx].tolist())
            if name != "inter":
                test_tmp_array = np.array(data[feat].loc[test_idx].tolist())
            else:
                test_tmp_array = np.array(self.test_inter[feat].tolist())

            np.save(self.dir / f"train_{feat}.npy", train_tmp_array)
            np.save(self.dir / f"valid_{feat}.npy", valid_tmp_array)
            np.save(self.dir / f"test_{feat}.npy", test_tmp_array)

            train_dict[feat] = train_tmp_array
            valid_dict[feat] = valid_tmp_array
            test_dict[feat] = test_tmp_array
            gc.collect()

        return train_dict, valid_dict, test_dict

    def _load_features(self, name: str, features: List[str]) -> Tuple[Dict]:
        """Load saved features of the dataset.

        Parameters
        ----------
        name: str
            The name of the dataset, should be one of {``user``, ``item``, ``inter``}.
        features: List[str]
            The features to be saved.

        Returns
        -------
        Tuple[Dict]
            The features of the dataset. (train, valid, test)
        """
        train_dict = {}
        valid_dict = {}
        test_dict = {}

        for feat in tqdm(features, f"Load {name} features"):
            train_path = self.dir / f"train_{feat}.npy"
            valid_path = self.dir / f"valid_{feat}.npy"
            test_path = self.dir / f"test_{feat}.npy"

            train_tmp_array = np.load(open(train_path, "rb"), allow_pickle=True)
            valid_tmp_array = np.load(open(valid_path, "rb"), allow_pickle=True)
            test_tmp_array = np.load(open(test_path, "rb"), allow_pickle=True)

            train_dict[feat] = train_tmp_array
            valid_dict[feat] = valid_tmp_array
            test_dict[feat] = test_tmp_array
            gc.collect()

        return train_dict, valid_dict, test_dict

    def train_test_split(
        self,
        test_num: int = None,
        train_start: Any = None,
        train_end: Any = None,
        test_start: Any = None,
        test_end: Any = None,
    ) -> None:
        """Split the dataset into train and test.

        If `test_num` is not None, the dataset will use the last `test_num` samples to be the test
        set and the rest to be the train set. Otherwise, the dataset will use the data
        between `train_start` and `train_end` to be the train set and the data between
        `test_start` and `test_end` to be the test set.

        Parameters
        ----------
        test_num: int, optional
            Use the last `test_num` samples to be the test set, by default ``None``.
        train_start: Any, optional
            The start time of the train set, by default ``None``.
        train_end: Any, optional
            The end time of the train set, by default ``None``.
        test_start: Any, optional
            The start time of the test set, by default ``None``.
        test_end: Any, optional
            The end time of the test set, by default ``None``.
        """
        # * filter users that have less than `num` interactions
        inter = self.data["inter"].copy()
        if test_num is not None:
            counter = inter.groupby([self.uid_name]).size().reset_index(name="count")
            valid_uid = counter.loc[counter["count"] > test_num][self.uid_name]
            inter = inter.loc[inter[self.uid_name].isin(valid_uid)]
            inter = inter.reset_index(drop=True)
            self.data["inter"] = inter

        # * split the dataset
        if test_num is not None:
            tmp = inter.groupby("user_id").apply(lambda x: list(x[:-test_num].index))
            inter_train_idx = np.concatenate(tmp.values)
            tmp = inter.groupby("user_id").apply(lambda x: list(x[-test_num:].index))
            inter_test_idx = np.concatenate(tmp.values)
        else:
            inter_train_idx = inter.loc[
                (inter[self.time_name] >= train_start)
                & (inter[self.time_name] < train_end)
            ].index
            inter_test_idx = inter.loc[
                (inter[self.time_name] >= test_start)
                & (inter[self.time_name] < test_end)
            ].index

        self.data["inter"] = inter.loc[inter_train_idx]

        self.test_label = np.array(
            inter.loc[inter_test_idx]
            .groupby(self.uid_name)[self.iid_name]
            .apply(list)
            .values.tolist()
        )
        self.test_inter = (
            inter.groupby(self.uid_name, as_index=False).last().reset_index(drop=True)
        )

    def train_valid_split(
        self,
        ratio: float = None,
        train_start: Any = None,
        train_end: Any = None,
        valid_start: Any = None,
        valid_end: Any = None,
        seed: int = 0,
    ) -> None:
        """Split the dataset into train and valid.

        If `ratio` is not None, the dataset will randomly smaple `ratio` of the whole
        dataset to be the valid set and the rest to be the train set. Otherwise, the
        dataset will use the data between `train_start` and `train_end` to be the
        train set and the data between `valid_start` and `valid_end` to be the valid
        set.

        Parameters
        ----------
        ratio : float, optional
            valid ratio of the whole dataset, by default ``None``.
        train_start : Any, optional
            The start time of the train set, by default ``None``.
        train_end : Any, optional
            The end time of the train set, by default ``None``.
        valid_start : Any, optional
            The start time of the valid set, by default ``None``.
        valid_end : Any, optional
            The end time of the valid set, by default ``None``.
        seed : int, optional
            Random seed when sampling fraction of the whole dataset, by default ``0``.
        """
        inter = self.data["inter"]
        if ratio is not None:
            inter_valid_idx = inter.sample(frac=ratio, random_state=seed).index
            inter_train_idx = inter.index.difference(inter_valid_idx)
        else:
            inter_train_idx = inter.loc[
                (inter[self.time_name] >= train_start)
                & (inter[self.time_name] < train_end)
            ].index
            inter_valid_idx = inter.loc[
                (inter[self.time_name] >= valid_start)
                & (inter[self.time_name] < valid_end)
            ].index

        self.index["train"] = inter_train_idx
        self.index["valid"] = inter_valid_idx

    def gen_dataset(
        self,
        user_feats: List[str],
        item_feats: List[str],
        inter_feats: List[str],
        test_candidates: Dict[int, List[int]] = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """Generate the dataset for training, validating and testing.

        Parameters
        ----------
        user_feats : List[str]
            The features of the user.
        item_feats : List[str]
            The features of the item.
        inter_feats : List[str]
            The features of the interaction.
        test_candidates : Dict[int, List[int]], optional
            The candidates items for each test user in ranking dataset, by default ``None``.
            key: user_id, value: list of item_id.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default ``True``.
        seed : int, optional
            Random seed when shuffling the dataset, by default ``0``.

        Raises
        ------
        ValueError
            If the task is rankindg and `test_candidates` is not provided.
        """
        if self.task == "ranking" and test_candidates is None:
            raise ValueError("`test_candidates` must be provided for ranking task.")

        self._check_features("user", user_feats)
        self._check_features("item", item_feats)
        self._check_features("inter", inter_feats)

        if test_candidates is not None:
            # * Regenerate the test set for ranking dataset
            candidate_num = len(list(test_candidates.values())[0])
            mask = self.test_inter[self.uid_name].isin(list(test_candidates.keys()))
            self.test_inter = self.test_inter[mask]
            self.test_inter = (
                self.test_inter.loc[self.test_inter.index.repeat(candidate_num)]
                .sort_values(by=self.uid_name)
                .reset_index(drop=True)
            )

            iid_values, p = self.test_inter[self.iid_name].values, 0
            for uid in tqdm(test_candidates, "Regenerate test set"):
                iid_values[p : p + candidate_num] = test_candidates[uid]
                p += candidate_num
            self.test_inter[self.iid_name] = iid_values

        if shuffle:
            self.data["inter"] = self.data["inter"].sample(frac=1, random_state=seed)
            self.data["inter"] = self.data["inter"].reset_index(drop=True)
        self._save_features("user", user_feats)
        self._save_features("item", item_feats)
        self._save_features("inter", inter_feats)

        with open(self.dir / "test_label.npy", "wb") as file:
            np.save(file, self.test_label)

    def load_dataset(
        self,
        user_feats: List[str],
        item_feats: List[str],
        inter_feats: List[str],
        batch_size: int,
        shuffle: bool = True,
    ) -> Tuple[tf.data.Dataset]:
        """Load saved dataset for training, validating and testing.

        Parameters
        ----------
        user_feats : List[str]
            The features of the user.
        item_feats : List[str]
            The features of the item.
        inter_feats : List[str]
            The features of the interaction.
        batch_size : int
            The batch size.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default True.

        Returns
        -------
        Tuple[tf.data.Dataset]
            The dataset for training, validating and testing.
        """
        user_train, user_valid, user_test = self._load_features("user", user_feats)
        item_train, item_valid, item_test = self._load_features("item", item_feats)
        inter_train, inter_valid, inter_test = self._load_features("inter", inter_feats)

        train_dict = {**user_train, **item_train, **inter_train}
        valid_dict = {**user_valid, **item_valid, **inter_valid}
        test_dict = {**user_test, **item_test, **inter_test}

        train_ds = tf.data.Dataset.from_tensor_slices(train_dict).batch(batch_size)
        valid_ds = tf.data.Dataset.from_tensor_slices(valid_dict).batch(batch_size)
        # test_ds = tf.data.Dataset.from_tensor_slices(test_dict)

        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(train_dict))

        test_label = np.load(self.dir / "test_label.npy")
        return train_ds, valid_ds, test_dict, test_label

    def get_feature_dim(
        self,
        user_features: List[str],
        item_features: List[str],
        inter_features: List[str],
    ) -> Dict[str, int]:
        """Generate a dictionary containing feature dimension info.

        Parameters
        ----------
        user_features : List[str]
            User feature list.
        item_features : List[str]
            Item feature list.
        inter_features : List[str]
            User-item interaction feature list.

        Returns
        -------
        Dict[str, int]
            A dictionary containing feature dimension info, ``{feature: dim}``.
        """
        feature_dim = {}
        for feat in user_features:
            try:
                feature_dim[feat] = np.max(self.data["user"][feat]) + 1
            except:
                pass
        for feat in item_features:
            try:
                feature_dim[feat] = np.max(self.data["item"][feat]) + 1
            except:
                pass
        for feat in inter_features:
            try:
                feature_dim[feat] = np.max(self.data["inter"][feat]) + 1
            except:
                pass

        return feature_dim

    # * ====================== To be implemented in the child class ======================

    def negative_sampling(self) -> None:
        """Negative sampling for the dataset, need to be implemented in the child class.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the child class.
        """
        raise NotImplementedError()


class PointWiseDataset(HandyRecDataset):
    """Dataset with point-wise inputs like ``uid, iid, label``

    The order of calling the methods should be:
        (0. `negative_sampling`)

        1. `train_test_split`

        2. `train_valid_split`

        3. `gen_dataset`

        4. `load_dataset`

    Attributes
    ----------
    task: str
        The task of the dataset, should be one of {``retrieval``, ``ranking``}.
    dir: Path
        The directory of the dataset.
    data: Dict
        The data of the dataset.
    uid_name: str
        The name of the user id column.
    iid_name: str
        The name of the item id column.
    inter_name: str
        The name of the interaction column.
    time_name: str
        The name of the time column.
    threshold: float
        The threshold for the interactions.
    label_name: str
        The name of the label.
    """

    def __init__(
        self,
        name: str,
        task: str,
        data: Dict,
        uid_name: str,
        iid_name: str,
        inter_name: str,
        time_name: str,
        label_name: str = "label",
        threshold: float = 0.0,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        name: str
            The name of the dataset.
        task: str
            The task of the dataset, should be one of {``retrieval``, ``ranking``}.
        data: Dict
            The data of the dataset.
        uid_name: str
            The name of the user id column.
        iid_name: str
            The name of the item id column.
        inter_name: str
            The name of the interaction column.
        time_name: str
            The name of the time column.
        label_name: str
            The name of the label column, by default ``"label"``.
        threshold: float, optional
            The threshold for the interactions, by default ``0.0``.
        """
        if label_name in data["inter"].columns:
            warnings.warn(
                "Predefined label column is not allowed, will be overwritten."
            )

        self.label_name = label_name
        data["inter"][self.label_name] = 1
        super().__init__(
            name, task, data, uid_name, iid_name, inter_name, time_name, threshold
        )

    def negative_sampling(self, neg_num: int) -> None:
        """Point-wise negative sampling for the dataset.

        Parameters
        ----------
        neg_num : int
            The number of negative samples for each positive sample.
        """
        inter = self.data["inter"]
        full_iid_set = set(inter[self.iid_name].unique())

        neg_inters = [0 for _ in range(inter[self.uid_name].nunique())]
        p = 0
        for uid, hist in tqdm(
            inter.groupby(self.uid_name), "Generate negative samples"
        ):
            # * Generate negative samples for each user
            pos_list = hist[self.iid_name].values
            neg_size = len(pos_list) * neg_num
            candidates = list(full_iid_set - set(pos_list))
            negs = np.random.choice(candidates, size=neg_size, replace=True)

            # * Generate data for negative samples
            neg_df = inter[inter[self.uid_name] == uid].copy()
            neg_df = pd.concat([neg_df] * neg_num, ignore_index=True)
            neg_df[self.iid_name] = negs
            neg_df[self.label_name] = 0

            # * Store the negative samples
            neg_inters[p] = neg_df
            p += 1

        inter = pd.concat([inter, *neg_inters], ignore_index=True)
        gc.collect()

        # * recalculate the index of train set and test set
        test_len = self.test_inter.shape[0]
        inter = pd.concat([inter, self.test_inter], ignore_index=True)
        self.index["train"] = inter.index[:-test_len]
        self.index["test"] = inter.index[-test_len:]

        self.data["inter"] = inter

    def gen_dataset(
        self,
        user_feats: List[str],
        item_feats: List[str],
        inter_feats: List[str],
        test_candidates: Dict[int, List[int]] = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """Generate the dataset for training, validating and testing.

        Parameters
        ----------
        user_feats : List[str]
            The features of the user.
        item_feats : List[str]
            The features of the item.
        inter_feats : List[str]
            The features of the interaction.
        test_candidates : Dict[int, List[int]], optional
            The candidates items for each test user in ranking dataset, by default ``None``.
            key: user_id, value: list of item_id.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default ``True``.
        seed : int, optional
            The seed for shuffling, by default ``0``.

        Raises
        ------
        ValueError
            If the task is rankindg and `test_candidates` is not provided.
        """
        inter_feats = list(set([self.label_name] + inter_feats))
        super().gen_dataset(
            user_feats, item_feats, inter_feats, test_candidates, shuffle, seed
        )

    def load_dataset(
        self,
        user_feats: List[str],
        item_feats: List[str],
        inter_feats: List[str],
        batch_size: int,
        shuffle: bool = True,
    ) -> Tuple[tf.data.Dataset]:
        """Load saved dataset for training, validating and testing.

        Parameters
        ----------
        user_feats : List[str]
            The features of the user.
        item_feats : List[str]
            The features of the item.
        inter_feats : List[str]
            The features of the interaction.
        batch_size : int
            The batch size.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default ``True``.

        Returns
        -------
        Tuple[tf.data.Dataset]
            The dataset for training, validating and testing.
        """
        inter_feats = list(set(inter_feats + [self.label_name]))

        user_train, user_valid, user_test = self._load_features("user", user_feats)
        item_train, item_valid, item_test = self._load_features("item", item_feats)
        inter_train, inter_valid, inter_test = self._load_features("inter", inter_feats)

        train_dict = {**user_train, **item_train, **inter_train}
        valid_dict = {**user_valid, **item_valid, **inter_valid}
        test_dict = {**user_test, **item_test, **inter_test}
        train_label = train_dict.pop(self.label_name)
        valid_label = valid_dict.pop(self.label_name)
        test_dict.pop(self.label_name)

        train_ds = tf.data.Dataset.from_tensor_slices((train_dict, train_label)).batch(
            batch_size
        )
        valid_ds = tf.data.Dataset.from_tensor_slices((valid_dict, valid_label)).batch(
            batch_size
        )
        # test_ds = tf.data.Dataset.from_tensor_slices(test_dict)
        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(train_dict))

        test_label = np.load(self.dir / "test_label.npy")
        return train_ds, valid_ds, test_dict, test_label


class PairWiseDataset(HandyRecDataset):
    """Dataset with pair-wise inputs like ``uid, iid, neg_iid``

    The order of calling the methods should be:
        (0. `negative_sampling`)

        1. `train_test_split`

        2. `train_valid_split`

        3. `gen_dataset`

        4. `load_dataset`

    Attributes
    ----------
    task: str
        The task of the dataset, should be one of {``retrieval``, ``ranking``}.
    dir: Path
        The directory of the dataset.
    data: Dict
        The data of the dataset.
    uid_name: str
        The name of the user id column.
    iid_name: str
        The name of the item id column.
    inter_name: str
        The name of the interaction column.
    time_name: str
        The name of the time column.
    threshold: float
        The threshold for the interactions.
    neg_iid_name: str
        The name of the negative item id column.
    """

    def __init__(
        self,
        name: str,
        task: str,
        data: Dict,
        uid_name: str,
        iid_name: str,
        inter_name: str,
        time_name: str,
        neg_iid_name: str,
        threshold: float = 0.0,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        name: str
            The name of the dataset.
        task: str
            The task of the dataset, should be one of {``retrieval``, ``ranking``}.
        data: Dict
            The data of the dataset.
        uid_name: str
            The name of the user id column.
        iid_name: str
            The name of the item id column.
        inter_name: str
            The name of the interaction column.
        time_name: str
            The name of the time column.
        neg_iid_name: str
            The name of the negative item id column.
        threshold: float, optional
            The threshold for the interactions, by default ``0.0``.
        """
        self.neg_iid_name = neg_iid_name
        super().__init__(
            name, task, data, uid_name, iid_name, inter_name, time_name, threshold
        )

    def gen_dataset(
        self,
        user_feats: List[str],
        item_feats: List[str],
        inter_feats: List[str],
    ) -> None:
        """Generate the dataset for training, validating and testing.

        Parameters
        ----------
        user_feats : List[str]
            The features of the user.
        item_feats : List[str]
            The features of the item.
        inter_feats : List[str]
            The features of the interaction.
        """
        inter_feats = list(set([self.neg_iid_name] + inter_feats))
        super().gen_dataset(user_feats, item_feats, inter_feats)


class SequenceWiseDataset(HandyRecDataset):
    """Dataset with sequence-wise inputs like ``uid, seq, neg_seq``

    The order of calling the methods should be:
        (0. `negative_sampling`)

        1. `train_test_split`

        2. `train_valid_split`

        3. `gen_dataset`

        4. `load_dataset`

    Attributes
    ----------
    task: str
        The task of the dataset, should be one of {``retrieval``, ``ranking``}.
    dir: Path
        The directory of the dataset.
    data: Dict
        The data of the dataset.
    uid_name: str
        The name of the user id column.
    iid_name: str
        The name of the item id column.
    inter_name: str
        The name of the interaction column.
    time_name: str
        The name of the time column.
    threshold: float
        The threshold for the interactions.
    seq_name: str
        The name of the sequence column.
    neg_seq_name: str
        The name of the negative sequence column.
    """

    def __init__(
        self,
        name: str,
        task: str,
        data: Dict,
        uid_name: str,
        iid_name: str,
        inter_name: str,
        time_name: str,
        seq_name: str,
        neg_seq_name: str,
        threshold: float = 0.0,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        name: str
            The name of the dataset.
        task: str
            The task of the dataset, should be one of {``retrieval``, ``ranking``}.
        data: Dict
            The data of the dataset.
        uid_name: str
            The name of the user id column.
        iid_name: str
            The name of the item id column.
        inter_name: str
            The name of the interaction column.
        time_name: str
            The name of the time column.
        seq_name: str
            The name of the sequence column.
        neg_seq_name: str
            The name of the negative sequence column.
        threshold: float, optional
            The threshold for the interactions, by default ``0.0``.
        """
        self.seq_name = seq_name
        self.neg_seq_name = neg_seq_name
        super().__init__(
            name, task, data, uid_name, iid_name, inter_name, time_name, threshold
        )

    def gen_dataset(
        self,
        user_feats: List[str],
        item_feats: List[str],
        inter_feats: List[str],
    ) -> None:
        """Generate the dataset for training, validating and testing.

        Parameters
        ----------
        user_feats : List[str]
            The features of the user.
        item_feats : List[str]
            The features of the item.
        inter_feats : List[str]
            The features of the interaction.
        """
        inter_feats = list(set([self.seq_name, self.neg_seq_name] + inter_feats))
        super().gen_dataset(user_feats, item_feats, inter_feats)
