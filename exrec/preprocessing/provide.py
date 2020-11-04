from dataclasses import asdict
from pathlib import Path

from exrec.core.dataframe import read_csv
from exrec.preprocessing.config import DataType
from exrec.preprocessing.dataset import RatingDatasetImpl
from exrec.preprocessing.interface import DataConfig, RatingDataset


def provide_dataset(
    data_path: Path, data_type: DataType, config: DataConfig, sep=","
) -> RatingDataset:
    data = read_csv(src_file=data_path, sep=sep)
    if data_type == DataType.RATING:
        return RatingDatasetImpl(data=data, **asdict(config))
    else:
        raise NotImplementedError("Not Implemented Dataset")
