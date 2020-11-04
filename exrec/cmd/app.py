from pathlib import Path

from exrec.cmd.config import AppConfig
from exrec.core.dataframe import read_csv
from exrec.core.label_encoder import provide_label_encoder
from exrec.core.logger import Logger, provide_logger
from exrec.evaluation.provide import provide_evaluator
from exrec.model.provide import provide_model
from exrec.model.repository import provide_repository
from exrec.preprocessing.provide import provide_dataset
from exrec.trainer.provide import provide_trainer
from exrec.usecase.interface import Usecase
from exrec.usecase.provide import provide_usecase


class App:
    usecase: Usecase
    logger: Logger

    def __init__(self, logger: Logger, usecase: Usecase):
        self.logger = logger
        self.usecase = usecase

        self.logger.info("Initialized App")

    def start(self) -> None:
        self.logger.info("Execute Usecase")
        self.usecase.execute(logger=self.logger.get_child(self.usecase.log_name()))
        self.logger.info("End the Usecase execution")


def provide_app(config: AppConfig) -> App:
    if config.out_dir.exists():
        write = (
            input(
                f"Would you like to overwirte files in the directory {config.out_dir}? Please type [Y/n]: "
            )
            == "Y"
        )
        if not write:
            raise FileExistsError(f"{config.out_dir} already exists")
    elif config.out_dir.is_file():
        raise RuntimeError(
            f"{config.out_dir} is expected to be a directory but it is a file."
        )
    else:
        config.out_dir.mkdir()

    if config.repo_dir.is_file():
        raise RuntimeError(
            f"{config.repo_dir} is expected to be a directory but it is a file."
        )
    elif not config.repo_dir.exists():
        config.repo_dir.mkdir()

    log_file = config.out_dir / Path("output.log")
    logger = provide_logger(name="App", output_file=log_file)

    ex_type = config.experiment_cfg

    model_repo = provide_repository(repo_dir=config.repo_dir)
    if config.model_load_file is not None:
        model = model_repo.load(config.model_load_file)
    else:
        model = provide_model(model_type=ex_type.model_type, config=config.model_cfg)

    logger.info(f"This model is {model.get_name()}")
    evaluator = provide_evaluator(
        evaluator_type=ex_type.evaluator_type, config=config.evaluator_cfg
    )
    trainer = provide_trainer(
        trainer_type=ex_type.trainer_type,
        config=config.trainer_cfg,
        logger=logger,
        model=model,
        evaluator=evaluator,
    )

    data_path = config.data_path
    all_data, train_data, test_data, valid_data = None, None, None, None

    user_enc, item_enc = None, None
    if config.unique_user is not None:
        unique_users = read_csv(src_file=config.unique_user)
        user_enc = provide_label_encoder(
            unique_users.unique(key=unique_users.columns[0])
        )
    elif config.user_one_idx:
        user_enc = provide_label_encoder(
            [i for i in range(1, config.data_cfg.n_users + 1)]
        )

    if config.unique_item is not None:
        unique_items = read_csv(src_file=config.unique_item)
        user_enc = provide_label_encoder(
            unique_items.unique(key=unique_items.columns[0])
        )
    elif config.item_one_idx:
        item_enc = provide_label_encoder(
            [i for i in range(1, config.data_cfg.n_items + 1)]
        )

    if data_path.all_data is not None:
        all_data = provide_dataset(
            data_path=data_path.all_data,
            data_type=ex_type.data_type,
            config=config.data_cfg,
        )
        all_data.label_encode(col=config.data_cfg.col_user, encoder=user_enc)
        all_data.label_encode(col=config.data_cfg.col_item, encoder=item_enc)

    if data_path.train_data is not None:
        train_data = provide_dataset(
            data_path=data_path.train_data,
            data_type=ex_type.data_type,
            config=config.data_cfg,
        )
        train_data.label_encode(col=config.data_cfg.col_user, encoder=user_enc)
        train_data.label_encode(col=config.data_cfg.col_item, encoder=item_enc)

    if data_path.test_data is not None:
        test_data = provide_dataset(
            data_path=data_path.test_data,
            data_type=ex_type.data_type,
            config=config.data_cfg,
        )
        test_data.label_encode(col=config.data_cfg.col_user, encoder=user_enc)
        test_data.label_encode(col=config.data_cfg.col_item, encoder=item_enc)

    if data_path.valid_data is not None:
        valid_data = provide_dataset(
            data_path=data_path.valid_data,
            data_type=ex_type.data_type,
            config=config.data_cfg,
        )
        valid_data.label_encode(col=config.data_cfg.col_user, encoder=user_enc)
        valid_data.label_encode(col=config.data_cfg.col_item, encoder=item_enc)

    score_file = config.out_dir / Path("score.csv")
    usecase = provide_usecase(
        usecase_type=ex_type.usecase_type,
        config=config.usecase_cfg,
        model=model,
        evaluator=evaluator,
        all_data=all_data,
        train_data=train_data,
        test_data=test_data,
        valid_data=valid_data,
        trainer=trainer,
        model_repo=model_repo,
        output_file=score_file,
        user_encoder=user_enc,
        item_encoder=item_enc,
    )

    return App(logger=logger, usecase=usecase)
