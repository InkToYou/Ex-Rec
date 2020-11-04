from dataclasses import asdict
from pathlib import Path
from typing import Optional

from exrec.core.interface import LabelEncoder
from exrec.evaluation.interface import Evaluator
from exrec.model.interface import Model, Repository
from exrec.preprocessing.interface import RatingDataset
from exrec.trainer.interface import Trainer
from exrec.usecase.config import UsecaseConfig, UsecaseType
from exrec.usecase.interface import Usecase
from exrec.usecase.train_test_eval import TrainTestEval


def provide_usecase(
    usecase_type: UsecaseType,
    config: UsecaseConfig,
    model: Model,
    evaluator: Evaluator,
    all_data: Optional[RatingDataset],
    train_data: Optional[RatingDataset],
    test_data: Optional[RatingDataset],
    valid_data: Optional[RatingDataset],
    trainer: Trainer,
    model_repo: Repository,
    output_file: Path,
    user_encoder: Optional[LabelEncoder],
    item_encoder: Optional[LabelEncoder],
) -> Usecase:
    if usecase_type == UsecaseType.TRAINTESTEVAL:
        return TrainTestEval(
            model=model,
            evaluator=evaluator,
            all_data=all_data,
            train_data=train_data,
            test_data=test_data,
            valid_data=valid_data,
            trainer=trainer,
            model_repo=model_repo,
            output_file=output_file,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            **asdict(config),
        )
    else:
        raise NotImplementedError("Not Implemented Usecase")
