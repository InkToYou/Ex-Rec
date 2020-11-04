from typing import Sequence

from sklearn import metrics


class MAE:
    def calc(self, true: Sequence[float], pred: Sequence[float]) -> float:
        return metrics.mean_absolute_error(true, pred)


class MSE:
    def calc(self, true: Sequence[float], pred: Sequence[float]) -> float:
        return metrics.mean_squared_error(true, pred, squared=True)


class RMSE:
    def calc(self, true: Sequence[float], pred: Sequence[float]) -> float:
        return metrics.mean_squared_error(true, pred, squared=False)
