from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Protocol, Sequence, Tuple

from exrec.core.id import ItemID, UserID
from exrec.core.interface import DataFrame, LabelEncoder


class SplitType(Enum):
    RANDOM = auto()
    SATISFIED = auto()
    TIMESTAMP = auto()


@dataclass
class DataConfig(metaclass=ABCMeta):
    n_users: int
    n_items: int
    col_user: str
    col_item: str


# DataFrame has (user, item, rating, (timestamp)) columns
class RatingDataset(Protocol):
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

    def __init__(
        self,
        data: DataFrame,
        n_users: int,
        n_items: int,
        col_user: str,
        col_item: str,
        col_rating: str,
        col_timestamp: Optional[str] = None,
    ):
        ...

    def log_name(self) -> str:
        ...

    def label_encode(self, col: str, encoder: Optional[LabelEncoder] = None) -> None:
        ...

    def label_decode(self, col: str, encoder: Optional[LabelEncoder] = None) -> None:
        ...

    def random_split(
        self, train_ratio: float, seed: int
    ) -> Tuple[RatingDataset, RatingDataset]:
        ...

    def satisfied_split(
        self,
        filter_by: str,
        train_ratio: float = 0.6,
        min_samples: int = 1,
        seed: int = 42,
    ) -> Tuple[RatingDataset, RatingDataset]:
        ...

    # Need the `timestamp` column in data
    def timestamp_split(
        self, train_ratio: float = 0.6
    ) -> Tuple[RatingDataset, RatingDataset]:
        ...

    def get_ratings(self) -> Tuple[Sequence[UserID], Sequence[ItemID], Sequence[float]]:
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

    def to_csv(self, file_name: str, index: bool) -> None:
        ...


# DataFrame has (user, item, type, (timestamp)) columns
class InteractionDataset(Protocol):
    cls_name: str

    data: DataFrame
    n_samples: int
    has_timestamp: bool
    n_users: int
    n_items: int
    col_user: str
    col_item: str
    col_type: str
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
    ):
        ...

    def weighted_count(self) -> RatingDataset:
        ...

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
