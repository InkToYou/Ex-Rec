from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict

from exrec.model.interface import ModelConfig


class ModelType(Enum):
    MF = auto()


@dataclass
class MFConfig(ModelConfig):
    n_users: int
    n_items: int
    dim: int
    reg: float
    epochs: int
    lr: float
    seed: int


type_to_cfg: Dict[ModelType, Callable[[Dict[str, Any]], ModelConfig]] = {
    ModelType.MF: lambda x: MFConfig(**x)
}
