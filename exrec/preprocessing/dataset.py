from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from exrec.core.id import ItemID, UserID
from exrec.core.interface import DataFrame, LabelEncoder
from exrec.preprocessing.interface import RatingDataset
from sklearn.model_selection import train_test_split


class RatingDatasetImpl:
    cls_name: str

    data: DataFrame
    n_samples: int
    has_timestamp: bool
    n_users: int
    n_items: int
    col_user: str
    col_item: str
    col_rating: str
    col_pred: Optional[str]
    col_timestamp: Optional[str]
    label_encoders: Dict[str, LabelEncoder]

    def __init__(
        self,
        data: DataFrame,
        n_users: int,
        n_items: int,
        col_user: str,
        col_item: str,
        col_rating: str,
        col_timestamp: Optional[str],
        col_pred: Optional[str],
    ):
        self.cls_name = self.__class__.__name__

        self.data = data.drop_duplicates()
        self.n_samples = len(data)
        self.n_users = n_users
        self.n_items = n_items
        self.has_timesamp = col_timestamp is not None
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_pred = None
        self.col_timestamp = col_timestamp
        self.id_cols = [col_user, col_item]
        self.split_by_col = {col_user: col_item, col_item: col_user}

        self.label_encoders: Dict[str, LabelEncoder] = {}

    def log_name(self) -> str:
        return self.cls_name

    def label_encode(self, col: str, encoder: Optional[LabelEncoder] = None) -> None:
        if col in self.label_encoders:
            raise RuntimeError("This column is already encoded.")

        if encoder is not None:
            self.data[col] = encoder.encode(self.data[col])
            self.label_encoders[col] = encoder

    def label_decode(self, col: str, encoder: Optional[LabelEncoder] = None) -> None:
        if col not in self.label_encoders:
            raise RuntimeError("This column is not encoded yet.")

        if encoder is not None:
            self.data[col] = encoder.decode(self.data[col])
            self.label_encoders.pop(col)

    def random_split(
        self, train_ratio: float = 0.6, seed: int = 42
    ) -> Tuple[RatingDataset, RatingDataset]:
        if train_ratio < 0 or train_ratio > 1:
            raise ValueError("train_ratio should be in [0, 1]")

        train, test = train_test_split(
            self.data.copy(deep=True).get_data(),
            train_size=train_ratio,
            test_size=None,
            shuffle=True,
            random_state=seed,
            straitify=None,
        )

        train = RatingDatasetImpl(
            data=type(self.data)(train),
            n_users=self.n_users,
            n_items=self.n_items,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_pred=self.col_pred,
            col_timestamp=self.col_timestamp,
        )

        test = RatingDatasetImpl(
            data=type(self.data)(test),
            n_users=self.n_users,
            n_items=self.n_items,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_pred=self.col_pred,
            col_timestamp=self.col_timestamp,
        )
        return train, test

    def satisfied_split(
        self,
        filter_by: str,
        train_ratio: float = 0.6,
        min_samples: int = 1,
        seed: int = 42,
    ) -> Tuple[RatingDataset, RatingDataset]:
        if train_ratio < 0 or train_ratio > 1:
            raise ValueError("train_ratio should be in [0, 1]")
        if filter_by not in self.id_cols:
            raise ValueError("filter_by should be 'user' or 'item'")
        if min_samples < 1:
            raise ValueError("min_samples should be integer greater than 0")

        if min_samples > 1:
            data = self.data.groupby(self.split_by_col[filter_by]).filter(
                lambda x: len(x) >= min_samples
            )
        else:
            data = self.data.copy(deep=True)

        train, test = train_test_split(
            data.get_data(),
            train_size=train_ratio,
            test_size=None,
            shuffle=True,
            straitify=data[filter_by],
            random_state=seed,
        )

        train = RatingDatasetImpl(
            data=type(self.data)(train),
            n_users=self.n_users,
            n_items=self.n_items,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_pred=self.col_pred,
            col_timestamp=self.col_timestamp,
        )

        test = RatingDatasetImpl(
            data=type(self.data)(test),
            n_users=self.n_users,
            n_items=self.n_items,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_pred=self.col_pred,
            col_timestamp=self.col_timestamp,
        )
        return train, test

    def timestamp_split(
        self, train_ratio: float = 0.6
    ) -> Tuple[RatingDataset, RatingDataset]:

        if not self.has_timestamp:
            raise ValueError("The DataFrame does not have the `timestamp` column")

        if train_ratio < 0 or train_ratio > 1:
            raise ValueError("train_ratio should be in [0, 1]")

        data = self.data.sort_values("timestamp")
        train, test = train_test_split(
            data.get_data(),
            train_size=train_ratio,
            test_size=None,
            shuffle=False,
            straitify=None,
        )

        train = RatingDatasetImpl(
            data=type(self.data)(train),
            n_users=self.n_users,
            n_items=self.n_items,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_pred=self.col_pred,
            col_timestamp=self.col_timestamp,
        )

        test = RatingDatasetImpl(
            data=type(self.data)(test),
            n_users=self.n_users,
            n_items=self.n_items,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_pred=self.col_pred,
            col_timestamp=self.col_timestamp,
        )
        return train, test

    def get_ratings(self) -> Tuple[Sequence[UserID], Sequence[ItemID], Sequence[float]]:
        return (
            self.data.values(self.col_user),
            self.data.values(self.col_item),
            self.data.values(self.col_rating),
        )

    def groupby_user(
        self,
    ) -> Dict[UserID, DataFrame]:
        return self.data.groupby(by=self.col_user).get_groups(
            [self.col_item, self.col_rating]
        )

    def get_data(self) -> DataFrame:
        return self.data

    def remove(self, remove: DataFrame) -> None:
        self.data = self.data.remove(remove=remove, keys=[self.col_user, self.col_item])

    def add(self, add: DataFrame) -> None:
        self.data = self.data.append(add=add)

    def to_csv(self, file_name: str, index: bool) -> None:
        self.data.to_csv(file_name=file_name, index=index)


# DataFrame has (user, item, type, (timestamp)) columns
class InteractionDatasetImpl:
    cls_name: str

    data: DataFrame
    n_samples: int
    has_timestamp: bool
    n_users: int
    n_items: int
    col_user: str
    col_item: str
    col_type: str
    col_weight: str
    col_timestamp: Optional[str]

    type_weight: Dict[str, int]

    def __init__(
        self,
        data: DataFrame,
        n_users: int,
        n_items: int,
        col_user: str,
        col_item: str,
        col_type: str,
        col_timestamp: Optional[str],
        type_weight: Dict[str, int],
        col_weight: str = "weight",
    ):
        self.cls_name = self.__class__.__name__

        self.data = data.drop_duplicates()
        self.n_samples = len(data)
        self.n_users = n_users
        self.n_items = n_items
        self.has_timestamp = col_timestamp is not None
        self.col_user = col_user
        self.col_item = col_item
        self.col_type = col_type
        self.col_timestamp = col_timestamp
        self.type_weight = type_weight
        self.col_weight = col_weight

    def weighted_count(self) -> RatingDataset:
        self.data.apply(
            f=lambda x: self.type_weight[x], key=self.col_type, dest=self.col_weight
        )
        w_data: DataFrame = self.data.groupby(
            by=[self.col_user, self.col_item]
        ).reduce_sum(key=self.col_weight)

        return RatingDatasetImpl(
            data=w_data,
            n_users=self.n_users,
            n_items=self.n_items,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_weight,
            col_timestamp=None,
            col_pred=None,
        )

    def negative_sampling(self) -> None:
        ...

    def groupby_user(
        self,
    ) -> Dict[UserID, DataFrame]:
        ...

    def get_data(self) -> DataFrame:
        ...

    def remove(self, remove: DataFrame) -> None:
        ...

    def add(self, add: DataFrame) -> None:
        ...

    def to_csv(self) -> None:
        ...
