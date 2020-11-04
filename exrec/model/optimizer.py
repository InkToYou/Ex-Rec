from typing import Sequence, Tuple

import numpy as np
from exrec.core.interface import Logger


def als(
    logger: Logger,
    users: Sequence[int],
    items: Sequence[int],
    ratings: Sequence[float],
    U: np.ndarray,
    V: np.ndarray,
    n_samples: int,
    dim: int,
    reg: float,
    lr: float,
    epochs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:

    np.random.seed(seed=seed)
    index = np.arange(n_samples)
    np.random.shuffle(index)

    for e in range(epochs):
        logger.info(f"Epoch: {e}/{epochs}")

        for idx in index:
            i = users[idx]
            j = items[idx]
            diff = ratings[idx] - np.dot(U[i, :], V[j, :])
            U[i, :] = U[i, :] + lr * (2 * diff * V[j, :] - reg * U[i, :])
            V[j, :] = V[j, :] + lr * (2 * diff * U[i, :] - reg * V[j, :])

    return (U, V)
