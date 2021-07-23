import argparse
import os
from super_image_resolution.utils import find_files, make_dirs
from super_image_resolution.sampler import RandomSampler, BaseSampleCondition
import numpy as np

from torchvision.transforms import ToPILImage


def check_if_file_dir(fp: str) -> bool:
    """
    Check if given file pointer is a directory or a file.

    Raises error if fp is neither a directory or a file.
    """
    if not os.path.isfile(fp) and not os.path.isdir(fp):
        msg = f"{fp} is neither a file or directory."
        raise argparse.ArgumentTypeError(msg)
    return fp


def check_str2bool(v: str) -> bool:
    """Convert standard string boolean flags to pythonic boolean values."""
    # https://stackoverflow.com/a/43357954/14916147
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(
    description="Run random sampling over svs files and save samples"
)
parser.add_argument(
    "-svs",
    action="store",
    help="path to svs file/s or directory containing svs file/s",
    type=check_if_file_dir,
    required=True,
)

parser.add_argument(
    "-samples_dir",
    action="store",
    help=("path to directory to store samples"),
    type=str,
    required=True,
)

parser.add_argument(
    "-recursive_svs_search",
    action="store",
    help="recursively search for svs files",
    type=check_str2bool,
    default=False,
)

parser.add_argument(
    "-create_samples_dir",
    action="store",
    help=("create directory to store samples"),
    type=check_str2bool,
    default=False,
)

parser.add_argument(
    "-num_samples_per_svs",
    action="store",
    help="number of samples per SVS file",
    type=int,
    default=1,
)

parser.add_argument(
    "-sample_size",
    action="store",
    help="dimension for each sample",
    type=int,
    default=256,
)

parser.add_argument(
    "-max_attempt_per_sample",
    action="store",
    help="Maximum number of attempts per sample",
    type=int,
    default=1,
)


args = parser.parse_args()

svs = args.svs
recursive_svs_search = args.recursive_svs_search
samples_dir = args.samples_dir
create_samples_dir = args.create_samples_dir
num_samples_per_svs = args.num_samples_per_svs
sample_size = args.sample_size
max_attempt_per_sample = args.max_attempt_per_sample

if os.path.isdir(svs) and not recursive_svs_search:
    raise ValueError(
        f"{svs} is a directory, " "try setting -recursive_svs_search flag to True."
    )
svs_files = find_files(svs, ext="svs", recursive=recursive_svs_search)

if not os.path.isdir(samples_dir) and not create_samples_dir:
    raise ValueError(
        f"{samples_dir} is not a directory or does not exists, "
        "try setting -create_samples_dir flag to True."
    )

if create_samples_dir:
    make_dirs(samples_dir, verbose=True)

print(f"Sampling from {len(svs_files)} SVS files")


class SampleCondition(BaseSampleCondition):
    """Sampling condition to accept or reject a sample."""

    def __call__(self, img: np.ndarray) -> bool:
        """Sample condition."""
        if img.mean() > 210:
            return False
        else:
            return True


sample_cond = SampleCondition()

for i, svs in enumerate(svs_files):
    sampler = RandomSampler(
        svs,
        scene_no=0,
        num_samples=num_samples_per_svs,
        sample_shape=sample_size,
        sample_cond=sample_cond,
        max_attempt_per_sample=max_attempt_per_sample,
        verbose=True,
        seed=None,
    )
    sampler.run()
    print(f"Sampled {len(sampler.samples)} from {svs}")
    for (x, y), sample in sampler.samples.items():
        fp = os.path.basename(svs).split(".")[0]
        fp = f"svs_{fp}_x{x}_y{y}"
        fp = os.path.join(samples_dir, f"{fp}.png")
        ToPILImage()(sample).save(fp)
        print(f"Saved {fp}", end="\r")
