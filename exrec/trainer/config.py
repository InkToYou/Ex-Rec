from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

from exrec.trainer.interface import TrainerConfig


class TrainerType(Enum):
    ONETIME = auto()


@dataclass
class OneTimeConfig(TrainerConfig):
    _dummy: Optional[Any] = None


type_to_cfg: Dict[TrainerType, Callable[[Dict[str, Any]], TrainerConfig]] = {
    TrainerType.ONETIME: lambda x: OneTimeConfig(**x)
}
