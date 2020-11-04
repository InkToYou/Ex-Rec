from abc import ABCMeta
from dataclasses import dataclass
from typing import Optional, Protocol

from exrec.core.interface import Logger
from exrec.evaluation.interface import Evaluator
from exrec.model.interface import Model
from exrec.preprocessing.interface import RatingDataset


@dataclass
class TrainerConfig(metaclass=ABCMeta):
    ...


class Trainer(Protocol):
    cls_name: str

    model: Model
    evaluator: Evaluator

    def log_name(self) -> str:
        ...

    def train(
        self,
        logger: Logger,
        train_data: RatingDataset,
        valid_data: Optional[RatingDataset] = None,
    ) -> None:
        ...
