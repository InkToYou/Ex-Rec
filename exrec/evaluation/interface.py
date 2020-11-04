from abc import ABCMeta
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence

from exrec.core.interface import Logger
from exrec.model.interface import Model
from exrec.preprocessing.interface import RatingDataset


@dataclass
class EvaluatorConfig(metaclass=ABCMeta):
    metrics: Sequence[Any]


class Metric(Protocol):
    def calc(self, true: Sequence[float], pred: Sequence[float]) -> float:
        ...


class Evaluator(Protocol):
    cls_name: str

    metrics: Sequence[Any]

    def log_name(self) -> str:
        ...

    def evaluate(
        self, logger: Logger, model: Model, test_data: RatingDataset
    ) -> Dict[str, float]:
        ...
