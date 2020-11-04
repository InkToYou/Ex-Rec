from typing import Any, Optional

from exrec.core.interface import Logger
from exrec.evaluation.interface import Evaluator
from exrec.model.interface import Model
from exrec.preprocessing.interface import RatingDataset


class OneTimeTrainer:
    cls_name: str

    model: Model
    evaluator: Evaluator

    def __init__(self, model: Model, evaluator: Evaluator, _dummy: Optional[Any]):
        self.cls_name = self.__class__.__name__

        self.model = model
        self.evaluator = evaluator

    def log_name(self) -> str:
        return self.cls_name

    def train(
        self,
        logger: Logger,
        train_data: RatingDataset,
        valid_data: Optional[RatingDataset] = None,
    ) -> None:

        logger.info("Start to train the model")
        self.model.fit(logger=logger.get_child(self.model.log_name()), data=train_data)
        logger.info("End to train the model")

        if valid_data is not None:
            logger.info("Get the score in the validate data")
            valid_score = self.evaluator.evaluate(
                logger=logger.get_child(name=self.evaluator.log_name()),
                model=self.model,
                test_data=valid_data,
            )
            logger.info(f"Validate scores: {valid_score}")
