from __future__ import annotations

from pathlib import Path
from typing import (Any, Callable, List, Optional, Protocol, Sequence, Type,
                    Union)


class Logger(Protocol):
    logger: Any

    def __init__(
        self,
        name: str,
        output_file: Optional[Path] = None,
        logger: Optional[Any] = None,
    ):
        ...

    def critical(self, message: str) -> None:
        ...

    def error(self, message: str) -> None:
        ...

    def warning(self, message: str) -> None:
        ...

    def info(self, message: str) -> None:
        ...

    def debug(self, message: str) -> None:
        ...

    def get_child(self, name: str) -> Logger:
        ...


class DataFrame(Protocol):
    data: Any

    def __init__(self, data):
        ...

    def __str__(self) -> str:
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, key: str) -> bool:
        ...

    def __iter__(self):
        ...

    def __next__(self):
        ...

    def __reversed__(self):
        ...

    def __getitem__(self, key: Union[str, List[str]]):
        ...

    def __setitem__(self, key: str, value) -> None:
        ...

    def __delitem__(self, key: str) -> None:
        ...

    @property
    def columns(self) -> List[str]:
        ...

    def sort_values(self, key: str) -> DataFrame:
        ...

    def groupby(self, by: Union[str, List[str]]) -> DataFrameGroupBy:
        ...

    def drop_duplicates(self) -> DataFrame:
        ...

    def copy(self, deep: bool) -> DataFrame:
        ...

    def get_data(self) -> Any:
        ...

    def values(self, key: str) -> Sequence[Any]:
        ...

    def remove(self, remove: DataFrame, keys: List[str]) -> DataFrame:
        ...

    def append(self, add: DataFrame) -> DataFrame:
        ...

    def unique(self, key: str) -> Sequence[Any]:
        ...

    def astype(self, key: str, type: Type) -> None:
        ...

    def apply(self, f: Callable[[Any], Any], key: str, dest: str) -> None:
        ...

    def to_csv(self, file_name: str, index: bool) -> None:
        ...


class DataFrameGroupBy(Protocol):
    group_data: Any

    def get_groups(self, keys: List[str]) -> dict:
        ...

    def filter(self, func: Callable[..., bool]) -> DataFrame:
        ...

    def get_dataframe(self) -> DataFrame:
        ...

    def reduce_sum(self, key: str) -> DataFrame:
        ...


class LabelEncoder(Protocol):
    encoder: Any

    def encode(self, labels: Sequence[Union[int, str]]) -> Sequence[int]:
        ...

    def decode(self, indices: Sequence[int]) -> Sequence[Union[int, str]]:
        ...
