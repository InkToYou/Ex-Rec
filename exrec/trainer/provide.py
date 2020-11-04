from dataclasses import asdict

from exrec.core.interface import Logger
from exrec.evaluation.interface import Evaluator
from exrec.model.interface import Model
from exrec.trainer.config import TrainerType
from exrec.trainer.interface import Trainer, TrainerConfig
from exrec.trainer.onetime import OneTimeTrainer


def provide_trainer(
    trainer_type: TrainerType,
    config: TrainerConfig,
    logger: Logger,
    model: Model,
    evaluator: Evaluator,
) -> Trainer:
    if trainer_type == TrainerType.ONETIME:
        return OneTimeTrainer(model=model, evaluator=evaluator, **asdict(config))
    else:
        raise NotImplementedError("Not Implemented Trainer.")
