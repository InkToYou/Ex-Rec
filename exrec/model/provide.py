from dataclasses import asdict

from exrec.model.config import ModelType
from exrec.model.factorization import MF as MFModel
from exrec.model.interface import Model, ModelConfig


def provide_model(model_type: ModelType, config: ModelConfig) -> Model:
    if model_type == ModelType.MF:
        model = MFModel(**asdict(config))
    else:
        raise NotImplementedError("Not Implemented Model")
    return model
