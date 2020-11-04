from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

from exrec.preprocessing.interface import SplitType
from exrec.usecase.interface import UsecaseConfig


class UsecaseType(Enum):
    TRAINTESTEVAL = auto()


@dataclass
class TrainTestEvalConfig(UsecaseConfig):
    train_ratio: Optional[float] = None
    split_type: Optional[SplitType] = None
    filter_by: Optional[str] = None
    min_samples: Optional[int] = None
    seed: Optional[int] = None


type_to_cfg: Dict[UsecaseType, Callable[[Dict[str, Any]], UsecaseConfig]] = {
    UsecaseType.TRAINTESTEVAL: lambda x: TrainTestEvalConfig(**x)
}
