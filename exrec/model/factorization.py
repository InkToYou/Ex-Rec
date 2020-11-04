from datetime import datetime

import numpy as np
from exrec.core.interface import Logger
from exrec.model.interface import Model
from exrec.model.optimizer import als
from exrec.preprocessing.interface import RatingDataset


class MF:
    cls_name: str

    # Basic information
    name: str
    n_users: int
    n_items: int

    # Hyper paramerters
    dim: int
    reg: float
    epochs: int
    lr: float
    seed: int

    # Learnable paramerters
    U: np.ndarray  # (n_users, dim)
    V: np.ndarray  # (n_items, dim)

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 20,
        reg: float = 0.1,
        epochs: int = 100,
        lr: float = 0.01,
        seed: int = 42,
    ):
        self.cls_name = self.__class__.__name__

        self.n_users = n_users
        self.n_items = n_items

        self.dim = dim
        self.reg = reg
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

        np.random.seed(seed=seed)
        self.U = np.random.rand(n_users, dim)
        self.V = np.random.rand(n_items, dim)

        self.set_name()

    def fit(self, logger: Logger, data: RatingDataset) -> Model:
        n_samples = data.n_samples
        users, items, ratings = data.get_ratings()
        logger.info("Start ALS")
        self.U, self.V = als(
            logger=logger.get_child(als.__name__),
            users=users,
            items=items,
            ratings=ratings,
            U=self.U,
            V=self.V,
            n_samples=n_samples,
            dim=self.dim,
            reg=self.reg,
            lr=self.lr,
            epochs=self.epochs,
            seed=self.seed,
        )
        logger.info("End ALS")

        return self

    def predict(
        self, logger: Logger, test_data: RatingDataset, col_pred: str = "pred"
    ) -> RatingDataset:
        grouped_data = test_data.groupby_user()
        pred_data = type(test_data.data)(
            {
                test_data.col_user: [],
                test_data.col_item: [],
                test_data.col_rating: [],
                col_pred: [],
            }
        )
        logger.info("Start the prediction.")
        for user, data in grouped_data.items():
            items = data.values(key=test_data.col_item)
            pred = np.dot(self.U[user, :], self.V[items, :].T)
            data[col_pred] = pred
            pred_data = pred_data.append(data)

        pred_data.astype(key=test_data.col_user, type=int)
        pred_data.astype(key=test_data.col_item, type=int)
        test_data.data = pred_data
        test_data.col_pred = col_pred
        logger.info("End the training")

        return test_data

    def log_name(self) -> str:
        return self.cls_name

    def set_name(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.name = f"MF:{now}-n_users:{self.n_users}-n_items{self.n_items}-dim:{self.dim}-reg{self.reg}-epochs:{self.epochs}-lr:{self.lr}-seed:{self.seed}"

    def get_name(self) -> str:
        return self.name
