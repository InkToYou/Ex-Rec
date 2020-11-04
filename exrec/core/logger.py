from logging import DEBUG, INFO, FileHandler, Formatter
from logging import Logger as LogLogger
from logging import StreamHandler, getLogger
from pathlib import Path
from typing import Optional

from exrec.core.interface import Logger


def provide_logger(
    name: str, output_file: Optional[Path] = None, logger: Optional[LogLogger] = None
) -> Logger:
    return LoggerImpl(name=name, output_file=output_file, logger=logger)


class LoggerImpl:
    logger: LogLogger

    def __init__(
        self,
        name: str,
        output_file: Optional[Path] = None,
        logger: Optional[LogLogger] = None,
    ):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = getLogger(name)

            self.logger.setLevel(DEBUG)
            formatter = Formatter(
                "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
            )

            stream_handler = StreamHandler()
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(INFO)
            self.logger.addHandler(stream_handler)

            if output_file is not None:
                file_handler = FileHandler(str(output_file))
                file_handler.setFormatter(formatter)
                file_handler.setLevel(DEBUG)
                self.logger.addHandler(file_handler)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def get_child(self, name: str) -> Logger:
        return LoggerImpl("", None, self.logger.getChild(name))
