from typing import Sequence, Union

from exrec.core.interface import LabelEncoder
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder


def provide_label_encoder(labels: Sequence[Union[int, str]]) -> LabelEncoder:
    return LabelEncoderImpl(labels=labels)


class LabelEncoderImpl:
    encoder: SkLabelEncoder

    def __init__(self, labels: Sequence[Union[int, str]]):
        self.encoder = SkLabelEncoder()
        self.encoder.fit(labels)

    def encode(self, labels: Sequence[Union[int, str]]) -> Sequence[int]:
        return self.encoder.transform(labels)

    def decode(self, indices: Sequence[int]) -> Sequence[Union[int, str]]:
        return self.encoder.inverse_transform(indices)
