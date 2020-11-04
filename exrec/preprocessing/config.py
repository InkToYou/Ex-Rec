from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

from exrec.preprocessing.interface import DataConfig


class DataType(Enum):
    RATING = auto()


@dataclass
class RatingDataConfig(DataConfig):
    n_users: int
    n_items: int
    col_user: str = 'user'
    col_item: str = 'item'
    col_rating: str = 'rate'
    col_pred: Optional[str] = None
    col_timestamp: Optional[str] = None


type_to_cfg: Dict[DataType, Callable[[Dict[str, Any]], DataConfig]] = {
    DataType.RATING: lambda x: RatingDataConfig(**x)
}
