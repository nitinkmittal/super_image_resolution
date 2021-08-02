"""Contains utility functions."""
import os
import pickle as pk
from pathlib import PosixPath
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchvision.utils import save_image
from tqdm import tqdm


def split_train_val(
    path: Union[PosixPath, str], num_train: Union[int, float], seed=None
) -> Tuple[List[Union[PosixPath, str]], List[Union[PosixPath, str]]]:
    """
    Split files inside given directory into train and validation set.

    Note: This function returns separate list of paths in train and
        validation set.
    """
    sample_paths = list(set(os.listdir(path)))
    sample_paths = np.sort(sample_paths)

    if isinstance(num_train, float):
        assert num_train > 0.0 and num_train < 1.0
        num_train = int(num_train * len(sample_paths))
    elif isinstance(num_train, int):
        assert num_train < len(os.listdir(path))
    else:
        raise ValueError(
            f"Expected num_train to be of type int or float,"
            f"got {type(num_train)}"
        )

    np.random.seed(seed)
    np.random.shuffle(sample_paths)
    train = sample_paths[:num_train]
    val = sample_paths[num_train:]
    assert len(set(train).intersection(val)) == 0
    return [os.path.join(path, sample) for sample in train], [
        os.path.join(path, sample) for sample in val
    ]


def split(
    path: Union[PosixPath, str], ratios: Sequence[float], seed=None
) -> Tuple[
    List[Union[PosixPath, str]],
    List[Union[PosixPath, str]],
    List[Union[PosixPath, str]],
]:
    """Split files inside given path into train, validation and test sets."""
    files = list(set(os.listdir(path)))
    files = np.sort(files)  # explicit sort required
    ratios = np.array(ratios)
    assert ratios.ndim == 1
    assert ratios.sum() == 1.0

    ratios = ratios.cumsum()
    np.random.seed(seed)
    np.random.shuffle(files)

    num_files = len(files)
    train = files[: int(ratios[0] * num_files)]
    val = files[int(ratios[0] * num_files) : int(ratios[1] * num_files)]
    test = files[int(ratios[1] * num_files) :]

    assert len(set(train).intersection(val)) == 0
    assert len(set(train).intersection(test)) == 0
    assert len(set(val).intersection(test)) == 0

    def make_file_paths(files: List[str]):
        return [os.path.join(path, f) for f in files]

    return make_file_paths(train), make_file_paths(val), make_file_paths(test)


def make_dirs(path: Union[str, PosixPath], verbose: bool = True):
    """Create directory if not present."""
    if os.path.isdir(path):
        if verbose:
            print(f"{path} already exists")
    else:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"Exception: {e} while creating {path}")
        else:
            if verbose:
                print(f"Created directory {path}")


def find_latest_model_version(path: Union[str, PosixPath]) -> int:
    """Find latest experiment version for a model."""
    if not os.path.isdir(path):
        raise ValueError(f"{path} does not exists")
    versions = os.listdir(path)
    versions = [v for v in versions if "version_" in v]
    if len(versions) == 0:
        return 0

    last_version = np.sort(versions)[-1]
    return int(last_version.replace("version_", ""))


def compute2d_means_and_stds(
    num_channels: int, data_loader
) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Compute means and stds for multi-channel 2d images."""
    means_in = torch.zeros(num_channels)
    stds_in = torch.zeros(num_channels)
    pixels_in = 0

    means_out = torch.zeros(num_channels)
    stds_out = torch.zeros(num_channels)
    pixels_out = 0

    for x, y in tqdm(data_loader, leave=False):
        means_in += x.sum(dim=(0, 2, 3))
        pixels_in += np.prod(np.array(x.shape)) // num_channels

        means_out += y.sum(dim=(0, 2, 3))
        pixels_out += np.prod(np.array(y.shape)) // num_channels

    means_in /= pixels_in
    means_out /= pixels_out

    means_in, means_out = means_in.view(1, num_channels, 1, 1), means_out.view(
        1, num_channels, 1, 1
    )
    for x, y in tqdm(data_loader, leave=False):
        stds_in += ((x - means_in) ** 2).sum(dim=(0, 2, 3))
        stds_out += ((y - means_out) ** 2).sum(dim=(0, 2, 3))

    stds_in /= pixels_in
    stds_out /= pixels_out

    return (
        (means_in.view(num_channels), stds_in.view(num_channels).sqrt()),
        (means_out.view(num_channels), stds_out.view(num_channels).sqrt()),
    )


def dump(fp: str, obj: object, verbose: bool = True):
    """Save python object as pickle."""
    try:
        with open(fp, "wb") as f:
            pk.dump(obj, f, protocol=pk.HIGHEST_PROTOCOL)
        if verbose:
            print(f"\nSuccessfully saved {fp}\n", end="\r")
    except Exception as e:
        print(f"Error: {e}")


def load(fp: str, verbose: bool = True) -> object:
    """Load pickle file as python object."""
    try:
        with open(fp, "rb") as f:
            obj = pk.load(f)
        if verbose:
            print(f"\nSuccessfully loaded {fp}\n", end="\r")
        return obj
    except Exception as e:
        print(f"Error: {e}")


def save_output(
    in_: Union[Tensor, List[Tensor]],
    out_true: Union[Tensor, List[Tensor]],
    out_pred: Union[Tensor, List[Tensor]],
    fp: Union[str, PosixPath],
    max_save: int = None,
    random_save: bool = True,
):
    """
    Save images as PNGs.

    Parameters
    ----------
    in_: input to model

    out_true: ground truth output

    out_pred: predicted output by model

    fp: filename

    max_save: maximum images to be saved

    random_save: to save random images
    """
    assert len(out_true) == len(out_pred)
    idxs = np.arange(len(out_true))
    if random_save:
        np.random.shuffle(idxs)

    if max_save is not None:
        max_save = min(len(out_true), max_save)
        idxs = idxs[:max_save]

    for i in idxs:
        save_image(
            tensor=[
                in_[i].to("cpu"),
                out_true[i].to("cpu"),
                out_pred[i].to("cpu"),
            ],
            fp=f"{fp}_sample_{i}.png",
            normalize=True,
        )


def find_files(path: str, ext: str, recursive: bool = True) -> List[str]:
    """Search and return files with extension."""
    abs_paths = []
    ext = ext.replace(".", "")

    def does_file_ext_match(path):
        fp = os.path.basename(path)
        if fp.split(".")[1] == ext:
            return True
        return False

    def recursive_search(path):
        if os.path.isfile(path) and does_file_ext_match(path):
            abs_paths.append(path)
        if os.path.isdir(path):
            for content in os.listdir(path):
                recursive_search(os.path.join(path, content))

    if recursive:
        recursive_search(path)
    else:
        if os.path.isfile(path) and does_file_ext_match(path):
            abs_paths.append(path)

    return list(set(abs_paths))


def load_torch_module(
    module, module_name: str, path: str, verbose: bool = True
):
    """Load Torch Module."""
    try:
        module.load_state_dict(torch.load(path))
    except Exception as e:
        print(f"Exception: {e}")
    if verbose:
        print(f"Loaded {module_name} from {path}", end="\r")
