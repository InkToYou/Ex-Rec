from abc import ABCMeta
from dataclasses import dataclass
from typing import Protocol

from exrec.core.interface import Logger


@dataclass
class UsecaseConfig(metaclass=ABCMeta):
    ...


class Usecase(Protocol):
    cls_name: str

    def log_name(self) -> str:
        ...

    def execute(self, logger: Logger) -> None:
        ...
