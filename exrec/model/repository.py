import pickle
from pathlib import Path

from exrec.model.interface import Model, Repository


class RepositoryImpl:
    repo_dir: Path

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir

    def store(self, model: Model) -> None:
        save_file = self.repo_dir / (model.get_name() + ".pkl")
        with open(str(save_file), "wb") as f:
            pickle.dump(model, f)

    def load(self, model_file: Path) -> Model:
        load_file = self.repo_dir / model_file
        with open(str(load_file), "rb") as f:
            model = pickle.load(f)
        return model


def provide_repository(repo_dir: Path) -> Repository:
    return RepositoryImpl(repo_dir=repo_dir)
