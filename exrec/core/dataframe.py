from pathlib import Path
from typing import Any, Callable, List, Sequence, Type, Union

import numpy as np
import pandas as pd
from exrec.core.interface import DataFrame, DataFrameGroupBy


class DataFrameImpl:
    data: pd.DataFrame

    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def __str__(self) -> str:
        return str(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __iter__(self):
        return iter(self.data)

    def __next__(self):
        return next(self.data)

    def __reversed__(self):
        return reversed(self.data)

    def __getitem__(self, key: Union[str, List[str]]):
        return self.data[key]

    def __setitem__(self, key: str, value) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    @property
    def columns(self) -> List[str]:
        return self.data.columns

    def sort_values(self, key: str) -> DataFrame:
        return DataFrameImpl(data=self.data.sort_values(key))

    def groupby(self, by: Union[str, List[str]]) -> DataFrameGroupBy:
        return DataFrameGroupByImpl(group_data=self.data.groupby(by=by))

    def drop_duplicates(self) -> DataFrame:
        return DataFrameImpl(data=self.data.drop_duplicates())

    def copy(self, deep: bool = True) -> DataFrame:
        return DataFrameImpl(data=self.data.copy(deep=deep))

    def get_data(self) -> pd.DataFrame:
        return self.data

    def values(self, key: str) -> np.ndarray:
        return self.data[key].values

    def remove(self, remove: DataFrame, keys: List[str]) -> DataFrame:
        return DataFrameImpl(
            data=self.data[~self.data[keys].isin(remove.data[keys]).all(axis=1)]
        )

    def append(self, add: DataFrame) -> DataFrame:
        return DataFrameImpl(data=self.data.append(add.get_data()))

    def unique(self, key: str) -> Sequence[Any]:
        return self.data[key].unique()

    def astype(self, key: str, type: Type) -> None:
        self.data[key] = self.data[key].astype(type)

    def apply(self, f: Callable[[Any], Any], key: str, dest: str) -> None:
        self.data[dest] = self.data[key].apply(f)

    def to_csv(self, file_name: str, index: bool) -> None:
        self.data.to_csv(file_name, index=False)


def read_csv(src_file: Path, sep: str = ",") -> DataFrame:
    data = pd.read_csv(src_file, sep=sep)
    return DataFrameImpl(data=data)


class DataFrameGroupByImpl:
    group_data: pd.core.groupby.GroupBy

    def __init__(self, group_data: pd.core.groupby.GroupBy):
        self.group_data = group_data

    def get_groups(self, keys: List[str]) -> dict:
        return {
            k: DataFrameImpl(data=v)
            for k, v in dict(tuple(self.group_data[keys])).items()
        }

    def filter(self, func: Callable[..., bool]) -> DataFrame:
        return DataFrameImpl(data=self.group_data.filter(func=func))

    def get_dataframe(self) -> DataFrame:
        return self.filter(lambda _: True)

    def reduce_sum(self, key: str) -> DataFrame:
        data = self.group_data[key].sum().reset_index()
        return DataFrameImpl(data=data)
