from pathlib import Path
from typing import Optional

from exrec.core.interface import LabelEncoder, Logger
from exrec.evaluation.interface import Evaluator
from exrec.model.interface import Model, Repository
from exrec.preprocessing.interface import RatingDataset, SplitType
from exrec.trainer.interface import Trainer


class TrainTestEval:
    cls_name: str

    # Config parameters
    filter_by: Optional[str]
    train_ratio: Optional[float]
    min_samples: Optional[int]
    seed: Optional[int]
    split_type: Optional[SplitType]

    # Basic parameters
    model: Model
    evaluator: Evaluator
    all_data: Optional[RatingDataset]
    train_data: Optional[RatingDataset]
    test_data: Optional[RatingDataset]
    valid_data: Optional[RatingDataset]
    model_repo: Repository
    output_file: Optional[Path]
    user_encoder: Optional[LabelEncoder]
    item_encoder: Optional[LabelEncoder]

    def __init__(
        self,
        model: Model,
        evaluator: Evaluator,
        trainer: Trainer,
        min_samples: Optional[int],
        filter_by: Optional[str],
        train_ratio: Optional[float],
        split_type: Optional[SplitType],
        seed: Optional[int],
        all_data: Optional[RatingDataset],
        train_data: Optional[RatingDataset],
        test_data: Optional[RatingDataset],
        valid_data: Optional[RatingDataset],
        model_repo: Repository,
        output_file: Optional[Path],
        user_encoder: Optional[LabelEncoder],
        item_encoder: Optional[LabelEncoder],
    ):
        self.cls_name = self.__class__.__name__

        self.filter_by = filter_by
        self.train_ratio = train_ratio
        self.min_samples = min_samples
        self.seed = seed
        self.split_type = split_type

        self.model = model
        self.evaluator = evaluator
        self.trainer = trainer

        self.all_data = all_data
        self.train_data = train_data
        self.test_data = test_data
        self.valid_data = valid_data

        self.model_repo = model_repo
        self.output_file = output_file

        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

    def log_name(self) -> str:
        return self.cls_name

    def execute(self, logger: Logger) -> None:

        if self.train_data is not None and self.test_data is not None:
            train = self.train_data
            test = self.test_data
            valid = self.valid_data
        else:
            if self.all_data is None:
                logger.error("all_data is not found.")
                raise RuntimeError("all_data is not found.")

            logger.info("Split data into train/test data.")
            if self.split_type is None:
                logger.error("split_type is not set.")
                raise RuntimeError("split_type is not set.")
            if self.seed is None:
                logger.error("seed is not set.")
                raise RuntimeError("seed is not set.")
            if self.train_ratio is None:
                logger.error("train_ratio is not set.")
                raise RuntimeError("train_ratio is not set.")

            valid = None
            if self.split_type == SplitType.RANDOM:
                train, test = self.all_data.random_split(
                    train_ratio=self.train_ratio, seed=self.seed
                )
            elif self.split_type == SplitType.SATISFIED:
                if self.filter_by is None or self.min_samples is None:
                    logger.error("filter_by and min_samples are not set.")
                    raise RuntimeError("filter_by and min_samples are not set.")
                train, test = self.all_data.satisfied_split(
                    filter_by=self.filter_by,
                    train_ratio=self.train_ratio,
                    min_samples=self.min_samples,
                    seed=self.seed,
                )
            elif self.split_type == SplitType.TIMESTAMP:
                train, test = self.all_data.timestamp_split(
                    train_ratio=self.train_ratio
                )
            else:
                logger.error("Not Implemented SplitType.")
                raise NotImplementedError("Not Implemented SplitType.")

        logger.info("Train the model")
        self.trainer.train(
            logger=logger.get_child(name=self.trainer.log_name()),
            train_data=train,
            valid_data=valid,
        )

        logger.info("Evaluate the model in test data")
        scores = self.evaluator.evaluate(
            logger=logger.get_child(name=self.evaluator.log_name()),
            model=self.model,
            test_data=test,
        )

        if self.output_file is not None:
            if self.user_encoder is not None:
                test.label_decode(col=test.col_user, encoder=self.user_encoder)
            if self.item_encoder is not None:
                test.label_decode(col=test.col_item, encoder=self.item_encoder)
            test.to_csv(str(self.output_file), index=False)

        logger.info("Save the model")
        self.model_repo.store(model=self.model)

        logger.info(f"Score in the test data: {scores}")
