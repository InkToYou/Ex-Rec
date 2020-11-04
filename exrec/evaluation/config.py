from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List

from exrec.evaluation.interface import EvaluatorConfig, Metric
from exrec.evaluation.metrics.ranking import AUC, DCG, NDCG
from exrec.evaluation.metrics.rating import MAE, MSE, RMSE


class EvaluatorType(Enum):
    SIMPLE = auto()


class MetricType(Enum):
    AUC = auto()
    DCG = auto()
    NDCG = auto()
    MAE = auto()
    MSE = auto()
    RMSE = auto()


GetMetric: Dict[MetricType, Metric] = {
    MetricType.AUC: AUC(),
    MetricType.DCG: DCG(),
    MetricType.NDCG: NDCG(),
    MetricType.MAE: MAE(),
    MetricType.MSE: MSE(),
    MetricType.RMSE: RMSE(),
}


@dataclass
class SimpleEvaluatorConfig(EvaluatorConfig):
    metrics: List[MetricType]

    def __post_init__(self):
        self.metrics = list(map(lambda x: MetricType[x], self.metrics))


type_to_cfg: Dict[EvaluatorType, Callable[[Dict[str, Any]], EvaluatorConfig]] = {
    EvaluatorType.SIMPLE: lambda x: SimpleEvaluatorConfig(**x)
}
