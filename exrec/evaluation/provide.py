from dataclasses import asdict

from exrec.evaluation.config import EvaluatorType
from exrec.evaluation.evaluate import SimpleEvaluator
from exrec.evaluation.interface import Evaluator, EvaluatorConfig


def provide_evaluator(
    evaluator_type: EvaluatorType, config: EvaluatorConfig
) -> Evaluator:
    if evaluator_type == EvaluatorType.SIMPLE:
        return SimpleEvaluator(**asdict(config))
    else:
        raise NotImplementedError("Not Implemented Evaluator.")
