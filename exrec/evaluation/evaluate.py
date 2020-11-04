from typing import Any, Dict, Sequence

from exrec.core.interface import Logger
from exrec.evaluation.config import GetMetric
from exrec.model.interface import Model
from exrec.preprocessing.interface import RatingDataset


class SimpleEvaluator:
    cls_name: str

    metrics: Sequence[Any]

    def __init__(self, metrics: Sequence[Any]):
        self.cls_name = self.__class__.__name__

        self.metrics = metrics

    def log_name(self) -> str:
        return self.cls_name

    def evaluate(
        self, logger: Logger, model: Model, test_data: RatingDataset
    ) -> Dict[str, float]:

        test_data = model.predict(
            logger=logger.get_child(model.log_name()), test_data=test_data
        )
        truth = test_data.get_data().values(key=test_data.col_rating)
        if test_data.col_pred is not None:
            pred = test_data.get_data().values(key=test_data.col_pred)
        else:
            logger.error("Prediction is failed.")
            raise RuntimeError("Prediction is failed.")

        score: Dict[str, float] = dict()
        for metric in self.metrics:
            score[metric.name] = GetMetric[metric].calc(true=truth, pred=pred)
        return score
