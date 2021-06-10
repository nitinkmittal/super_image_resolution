import os
from typing import List, Tuple, Union

import numpy as np
from pathlib import PosixPath


def split(
    path: Union[PosixPath, str], num_train: Union[int, float], seed=None
) -> Tuple[List[Union[PosixPath, str]], List[Union[PosixPath, str]]]:
    """Split files inside given directory into train and validation set."""
    sample_paths = list(set(os.listdir(path)))

    if isinstance(num_train, float):
        assert num_train > 0.0 and num_train < 1.0
        num_train = int(num_train * len(sample_paths))
    elif isinstance(num_train, int):
        assert num_train < len(os.listdir(path))
    else:
        raise ValueError(
            f"Expected num_train to be of type int or float, got {type(num_train)}"
        )

    np.random.seed(seed)
    np.random.shuffle(sample_paths)
    train = sample_paths[:num_train]
    val = sample_paths[num_train:]
    assert len(set(train).intersection(val)) == 0
    return [os.path.join(path, sample) for sample in train], [
        os.path.join(path, sample) for sample in val
    ]
