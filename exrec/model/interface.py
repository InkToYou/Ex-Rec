from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from exrec.core.interface import Logger
from exrec.preprocessing.interface import RatingDataset


@dataclass
class ModelConfig(metaclass=ABCMeta):
    pass


class Model(Protocol):
    cls_name: str

    # This name is used when storing the model to a model repository.
    name: str

    def fit(self, logger: Logger, data: RatingDataset) -> Model:
        ...

    def predict(
        self,
        logger: Logger,
        test_data: RatingDataset,
        col_pred: str = "pred",
    ) -> RatingDataset:
        ...

    def log_name(self):
        ...

    def set_name(self) -> None:
        ...

    def get_name(self) -> str:
        ...


class Repository(Protocol):
    repo_dir: Path

    def __init__(self, repo_dir: Path):
        ...

    def store(self, model: Model) -> None:
        ...

    def load(self, model_file: Path) -> Model:
        ...
