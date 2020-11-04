from typing import Sequence

from sklearn import metrics


class AUC:
    def calc(self, true: Sequence[float], pred: Sequence[float]) -> float:
        return metrics.roc_auc_score(true, pred)


class DCG:
    def calc(self, true: Sequence[float], pred: Sequence[float]) -> float:
        return metrics.dcg_score(true, pred)


class NDCG:
    def calc(self, true: Sequence[float], pred: Sequence[float]) -> float:
        return metrics.ndcg_score(true, pred)
