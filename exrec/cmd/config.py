from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from exrec.evaluation.config import EvaluatorType
from exrec.evaluation.config import type_to_cfg as type_to_evaluator_cfg
from exrec.evaluation.interface import EvaluatorConfig
from exrec.model.config import ModelType
from exrec.model.config import type_to_cfg as type_to_model_cfg
from exrec.model.interface import ModelConfig
from exrec.preprocessing.config import DataType
from exrec.preprocessing.config import type_to_cfg as type_to_data_cfg
from exrec.preprocessing.interface import DataConfig, SplitType
from exrec.trainer.config import TrainerType
from exrec.trainer.config import type_to_cfg as type_to_trainer_cfg
from exrec.trainer.interface import TrainerConfig
from exrec.usecase.config import UsecaseConfig, UsecaseType
from exrec.usecase.config import type_to_cfg as type_to_usecase_cfg


@dataclass
class DataPath:
    all_data: Optional[Path] = None
    train_data: Optional[Path] = None
    test_data: Optional[Path] = None
    valid_data: Optional[Path] = None


@dataclass
class ExperimentConfig:
    usecase_type: UsecaseType
    data_type: DataType
    model_type: ModelType
    trainer_type: TrainerType
    evaluator_type: EvaluatorType


@dataclass
class AppConfig:
    experiment_cfg: ExperimentConfig
    usecase_cfg: UsecaseConfig
    data_cfg: DataConfig
    model_cfg: ModelConfig
    evaluator_cfg: EvaluatorConfig
    trainer_cfg: TrainerConfig
    data_path: DataPath
    out_dir: Path
    repo_dir: Path
    user_one_idx: bool = False
    item_one_idx: bool = False
    unique_user: Optional[Path] = None
    unique_item: Optional[Path] = None
    model_load_file: Optional[Path] = None


base_type_hooks: Dict[Any, Callable[[Any], Any]] = {
    SplitType: lambda x: SplitType[x],
    DataType: lambda x: DataType[x],
    EvaluatorType: lambda x: EvaluatorType[x],
    ModelType: lambda x: ModelType[x],
    TrainerType: lambda x: TrainerType[x],
    UsecaseType: lambda x: UsecaseType[x],
    Optional[Path]: Path,
    Path: Path,
}


def provide_type_hooks(ex_cfg: ExperimentConfig) -> Dict[Any, Callable[[Any], Any]]:

    base_type_hooks[UsecaseConfig] = type_to_usecase_cfg[ex_cfg.usecase_type]
    base_type_hooks[DataConfig] = type_to_data_cfg[ex_cfg.data_type]
    base_type_hooks[ModelConfig] = type_to_model_cfg[ex_cfg.model_type]
    base_type_hooks[TrainerConfig] = type_to_trainer_cfg[ex_cfg.trainer_type]
    base_type_hooks[EvaluatorConfig] = type_to_evaluator_cfg[ex_cfg.evaluator_type]

    return base_type_hooks
