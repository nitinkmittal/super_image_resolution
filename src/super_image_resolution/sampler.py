"""Contain Sampler to sample from SVS Whole Slide Images."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import slideio


class BaseSampleCondition(ABC):
    @abstractmethod
    def __call__(self, img: np.ndarray) -> bool:
        pass


class RandomSampler:
    """Randomly sample patches from SVS file."""

    def __init__(
        self,
        fp: str,
        scene_no: int,
        sample_shape: Union[int, Tuple[int, int]],
        num_samples: int,
        sample_cond: BaseSampleCondition = None,
        max_attempt_per_sample: int = 1,
        channel_indices: List[int] = [],
        verbose: bool = False,
        seed: int = None,
    ):

        """
        Parameters
        ----------
        fp: filepath

        scene_no: scene/tile number to sample from

        shape: sample shape

        num_samples: number of samples

        sample_cond: sampling condition to accept or reject sample

        max_attempt_per_sample: maximum number of attempts for each sample

        channel_indices: color channels to be retrieved

        verbose: verbosity

        seed: controls randomness
        """
        self.slide = slideio.open_slide(fp, "SVS")
        self.scene = self.slide.get_scene(scene_no)
        self.W, self.H = self.scene.rect[-2:]
        self.sample_shape = sample_shape
        self._validate_sample_shape()

        if isinstance(self.sample_shape, int):
            self.w = self.h = self.sample_shape
        else:
            self.w, self.h = self.sample_shape
        self.X, self.Y = (self.W - self.w - 1, self.H - self.h - 1)
        self._validate_X()
        self._validate_Y()
        self.channel_indices = channel_indices
        self.num_samples = num_samples
        self.max_attempts = max_attempt_per_sample * self.num_samples
        self.sample_cond = sample_cond
        self._validate_sample_cond()
        self.coords = set()
        self.samples = {}
        self.seed = seed
        self._validate_seed()
        self.verbose = verbose

    def _validate_sample_shape(self):
        """Validate shape."""
        if not isinstance(self.sample_shape, (tuple, int)):
            raise ValueError(
                f"Expected sample_shape of type int or tuple, "
                f"got {type(self.sample_shape)}"
            )

    def _validate_X(self):
        """Validate X."""
        if self.X < 0:
            raise ValueError(
                f"Sample width: {self.w}, greater than slide width: {self.W}, "
                f"{self.W}-{self.w}-1 < 0, try passing smaller sample width"
            )

    def _validate_Y(self):
        """Validate Y."""
        if self.Y < 0:
            raise ValueError(
                f"Sample height: {self.h}, greater than slide height: {self.H}, "
                f"{self.H}-{self.h}-1 < 0, try passing smaller sample height"
            )

    def _validate_sample_cond(self):
        if self.sample_cond is not None:
            if not isinstance(self.sample_cond, BaseSampleCondition):
                raise ValueError(
                    f"Sampling condition should be instance of BaseSampleCondition, "
                    f"got {type(self.sample_cond)}"
                )

    def _validate_seed(
        self,
    ):
        """Validate seed."""
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(
                f"Expected seed of int type, got {type(self.seed)}"
            )

    def _update_seed(
        self,
    ):
        """Update seed."""
        if self.seed is not None:
            self.seed += 1

    def _sample_x_y(self):
        """Sample x and y."""
        np.random.seed(self.seed)
        return (np.random.choice(self.X), np.random.choice(self.Y))

    def _pick_img(self, img) -> bool:
        """Decide to accept or reject image."""
        if self.sample_cond is None:
            return True
        return self.sample_cond(img)

    def _printer(self, num_attempts: int):
        """Print sampler progress."""
        if self.verbose:
            print(
                f"Attempt: {num_attempts}/{self.max_attempts}, "
                f" sample number: {len(self.samples)}/{self.num_samples}",
            )

    def run(self):
        """Run sampling."""
        num_attempts = 0

        while num_attempts <= self.max_attempts and (
            len(self.samples) < self.num_samples
        ):
            num_attempts += 1

            self._update_seed()
            x, y = self._sample_x_y()

            if len(self.coords.intersection([(x, y)])) > 0:
                continue
            else:
                self.coords.add((x, y))
                img = self.scene.read_block(
                    rect=(x, y, self.w, self.h),
                    channel_indices=self.channel_indices,
                )
                if self._pick_img(img):
                    self.samples[(x, y)] = img

            self._printer(num_attempts)
