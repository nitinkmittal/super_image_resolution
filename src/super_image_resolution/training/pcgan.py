import os
from copy import deepcopy
from pathlib import PosixPath
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.utils import save_image
from tqdm import tqdm

from super_image_resolution.modules.patch_discriminator import (
    PatchDiscriminator,
)
from super_image_resolution.modules.unet import UNet
from super_image_resolution.utils import (
    dump,
    load,
    load_torch_module,
    make_dirs,
)


class PCGAN:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gen_norm: str,
        gen_act: str,
        gen_final_act: str,
        dis_norm: str,
        dis_act: str,
        dis_final_act: str,
        dis_depth: int,
        dis_kernel_size: Union[int, Tuple[int, int]],
        dis_stride: Union[int, Tuple[int, int]],
        dis_padding: Union[int, Tuple[int, int]],
        gen_loss_weight: float = 1.0,
        dis_loss_weight: float = 1.0,
        dis_lr: float = 1e-3,
        gen_lr: float = 1e-3,
        gen_step_interval: int = 1,
        dis_step_interval: int = 1,
        val_step_interval: int = 1,
        val_output_save_interval: int = 1,
        max_val_batches: int = -1,
        max_num_val_output_save: int = 1,
        device: torch.device = torch.device("cpu"),
        experiment_version: int = 0,
        experiment_save_path: str = None,
        modules_dir: str = None,
        metrics_dir: str = None,
        metrics_save_interval: int = 1,
        verbose: bool = True,
        **kwargs,
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.gen_norm = gen_norm
        self.gen_lr = gen_lr
        self.gen_act = gen_act
        self.gen_final_act = gen_final_act
        self.gen_norm = gen_norm
        self.gen_loss_weight = gen_loss_weight
        self.gen_step_interval = gen_step_interval
        self.dis_act = dis_act
        self.dis_norm = dis_norm
        self.dis_lr = dis_lr
        self.dis_final_act = dis_final_act
        self.dis_depth = dis_depth
        self.dis_kernel_size = dis_kernel_size
        self.dis_stride = dis_stride
        self.dis_padding = dis_padding
        self.dis_loss_weight = dis_loss_weight
        self.dis_step_interval = dis_step_interval
        self.val_step_interval = val_step_interval
        self.val_output_save_interval = val_output_save_interval
        self.max_val_batches = max_val_batches
        self.max_num_val_output_save = max_num_val_output_save
        self.verbose = verbose
        self._train_step = 0
        self._val_step = 0
        self.experiment_version = experiment_version
        self.experiment_save_path = experiment_save_path
        self.modules_dir = modules_dir
        self.metrics_dir = metrics_dir
        self.metrics_save_interval = metrics_save_interval

        self.hparams = locals()
        del self.hparams["self"]

        self.gen_fp = "gen.pth"
        self.dis_fp = "dis.pth"
        self.gen_optim_fp = "gen_optim.pth"
        self.dis_optim_fp = "dis_optim.pth"
        self.hparams_fp = "hparams.p"
        self.metrics_fp = "metrics.p"

        if self.experiment_save_path is None:
            self.experiment_save_path = os.getcwd()

        self.experiment_save_path = os.path.join(
            self.experiment_save_path, f"version_{self.experiment_version}"
        )
        make_dirs(self.experiment_save_path, verbose=self.verbose)
        self.hparams["experiment_save_path"] = self.experiment_save_path

        self.hparams_dir = os.path.join(self.experiment_save_path, "hparams")
        make_dirs(self.hparams_dir, verbose=self.verbose)
        self.hparams_path = os.path.join(self.hparams_dir, self.hparams_fp)
        self.hparams["hparams_path"] = self.hparams_path

        self.val_results_dir = os.path.join(self.experiment_save_path, "val")
        make_dirs(self.val_results_dir, verbose=self.verbose)
        self.hparams["val_results_dir"] = self.val_results_dir

        self.gen = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            norm=self.gen_norm,
            activation=self.gen_act,
            final_act=self.gen_final_act,
        ).to(self.device)

        self.dis = PatchDiscriminator(
            in_channels=self.out_channels,
            depth=self.dis_depth,
            kernel_size=self.dis_kernel_size,
            stride=self.dis_stride,
            padding=self.dis_padding,
            activation=self.dis_act,
            norm=self.dis_norm,
            final_activation=self.dis_final_act,
        ).to(self.device)

        self.gen_optim = torch.optim.Adam(
            self.gen.parameters(), lr=self.gen_lr
        )
        self.dis_optim = torch.optim.Adam(
            self.dis.parameters(), lr=self.dis_lr
        )

        self.reconst_criterion = nn.L1Loss()
        self.adv_criterion = nn.BCEWithLogitsLoss()

        self._load_modules_if_exist()
        self._load_metrics_if_exist()
        dump(fp=self.hparams_path, obj=self.hparams, verbose=self.verbose)

    def _init_metrics(self):
        """Initialize traceable loss metrics."""
        losses = {
            "gen_reconst_loss": [],
            "gen_adv_loss": [],
            "dis_adv_loss": [],
        }
        self.metrics = {
            "train": deepcopy(losses),
            "val": deepcopy(losses),
            "min_val_loss": np.inf,
        }

    def _load_modules_if_exist(self):
        """Load modules if already exist."""
        if self.modules_dir is not None:
            fps = os.listdir(self.modules_dir)
            if self.gen_fp in fps:
                load_torch_module(
                    module=self.gen,
                    module_name="Generator",
                    path=os.path.join(self.modules_dir, self.gen_fp),
                    verbose=self.verbose,
                )
            if self.dis_fp in fps:
                load_torch_module(
                    module=self.dis,
                    module_name="Discriminator",
                    path=os.path.join(self.modules_dir, self.dis_fp),
                    verbose=self.verbose,
                )
            if self.gen_optim_fp in fps:
                load_torch_module(
                    module=self.gen_optim,
                    module_name="Generator optimizer",
                    path=os.path.join(self.modules_dir, self.gen_optim_fp),
                    verbose=self.verbose,
                )
            if self.dis_optim_fp in fps:
                load_torch_module(
                    module=self.dis_optim,
                    module_name="Discriminator optimizer",
                    path=os.path.join(self.modules_dir, self.dis_optim_fp),
                    verbose=self.verbose,
                )
        self.modules_dir = os.path.join(self.experiment_save_path, "modules")
        make_dirs(self.modules_dir, verbose=self.verbose)
        self.gen_path = os.path.join(self.modules_dir, self.gen_fp)
        self.dis_path = os.path.join(self.modules_dir, self.dis_fp)
        self.gen_optim_path = os.path.join(self.modules_dir, self.gen_optim_fp)
        self.dis_optim_path = os.path.join(self.modules_dir, self.dis_optim_fp)
        self.hparams["modules_dir"] = self.modules_dir
        self.hparams["gen_path"] = self.gen_path
        self.hparams["dis_path"] = self.dis_path
        self.hparams["gen_optim_path"] = self.gen_optim_path
        self.hparams["dis_optim_path"] = self.dis_optim_path

    def _load_metrics_if_exist(self):
        """Load metrics if already exist."""
        if self.metrics_dir is not None and self.metrics_fp in os.listdir(
            self.metrics_dir
        ):
            self.metrics = load(
                fp=os.path.join(self.metrics_dir, self.metrics_fp),
                verbose=self.verbose,
            )
        else:
            self._init_metrics()

        self.metrics_dir = os.path.join(self.experiment_save_path, "metrics")
        make_dirs(self.metrics_dir, verbose=self.verbose)
        self.metrics_path = os.path.join(self.metrics_dir, self.metrics_fp)
        self.hparams["metrics_dir"] = self.metrics_dir
        self.hparams["metrics_path"] = self.metrics_path

    def forward(self, x: Tensor, mode="train") -> Tensor:
        """Perform forward pass."""
        with torch.set_grad_enabled(True if mode == "train" else False):
            return self.gen(x)

    def _save_metrics(self, force_save: bool = False):
        """Save metrics."""
        if (self._train_step) % self.metrics_save_interval == 0 or force_save:
            dump(fp=self.metrics_path, obj=self.metrics, verbose=self.verbose)

    def _add_loss_in_metrics(self, loss: float, type_: str, mode: str):
        """Save loss in metrics."""
        self.metrics[mode][type_].append([self._train_step, loss])

    def _save_modules_if_min_val_loss(
        self,
        losses: List[Tuple[int, float]],
    ):
        """Save torch modules if loss is lesser than min_val_loss."""
        loss = np.mean([loss for _, loss in losses])
        if loss < self.metrics["min_val_loss"]:
            if self.verbose:
                print(
                    "Minimum validation loss found, "
                    f"step: {self._train_step}, loss: {loss}",
                    end="\r",
                )
            self.metrics["min_val_loss"] = loss
            torch.save(self.gen.state_dict(), self.gen_path)
            torch.save(self.gen_optim.state_dict(), self.gen_optim_path)
            torch.save(self.dis.state_dict(), self.dis_path)
            torch.save(self.dis_optim.state_dict(), self.dis_optim_path)

    def _save_val_outputs(
        self,
        fps: Tuple[Union[PosixPath, str]],
        X: Tensor,
        Y: Tensor,
        Y_hat: Tensor,
    ):
        """Save generator outputs from validation step."""
        for i, (fp, x, y, y_hat) in enumerate(zip(fps, X, Y, Y_hat)):
            fp = (
                f"{os.path.basename(fp).split('.')[0]}_"
                f"train_step_{self._train_step}.png"
            )
            save_image(
                tensor=[x, y, y_hat],
                fp=os.path.join(self.val_results_dir, fp),
                pad_value=1.0,
            )
            if (i + 1) >= self.max_num_val_output_save:
                break

    def _generator_step(
        self, x: Tensor, y: Tensor, mode: str
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass w.r.t to Generator."""
        y_hat = self.gen(x)

        reconst_loss = self.gen_loss_weight * self.reconst_criterion(y_hat, y)
        self._add_loss_in_metrics(
            loss=reconst_loss.item(),
            type_="gen_reconst_loss",
            mode=mode,
        )

        dis_hat_out = self.dis(y_hat)
        adv_loss = self.dis_loss_weight * self.adv_criterion(
            dis_hat_out,
            torch.ones_like(
                dis_hat_out, requires_grad=False, device=self.device
            ),
        )
        self._add_loss_in_metrics(
            loss=adv_loss.item(),
            type_="gen_adv_loss",
            mode=mode,
        )

        return y_hat, reconst_loss + adv_loss

    def _discriminator_step(self, x: Tensor, y: Tensor, mode: str) -> Tensor:
        """Forward pass w.r.t Discriminator."""
        y_hat = self.gen(x)
        dis_hat_out = self.dis(y_hat.detach())
        adv_hat_loss = self.dis_loss_weight * self.adv_criterion(
            dis_hat_out,
            torch.zeros_like(
                dis_hat_out, requires_grad=False, device=self.device
            ),
        )

        dis_out = self.dis(y)
        adv_loss = self.dis_loss_weight * self.adv_criterion(
            dis_out,
            torch.ones_like(dis_out, requires_grad=False, device=self.device),
        )

        self._add_loss_in_metrics(
            loss=(adv_hat_loss + adv_loss).item(),
            type_="dis_adv_loss",
            mode=mode,
        )
        return adv_hat_loss + adv_loss

    def _training_step(
        self, batch: Tuple[List[Union[PosixPath, str]], Tensor, Tensor]
    ):
        """Train Generator and Discriminator."""
        self._train_step += 1
        _, x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        gen_update = self._train_step % self.gen_step_interval == 0
        dis_update = self._train_step % self.dis_step_interval == 0

        with torch.set_grad_enabled(True):
            if gen_update:
                self.gen_optim.zero_grad()
                _, loss = self._generator_step(x, y, mode="train")
                loss.backward()
                self.gen_optim.step()

            if dis_update:
                self.dis_optim.zero_grad()
                loss = self._discriminator_step(x, y, mode="train")
                loss.backward()
                self.dis_optim.step()

    def _validation_step(
        self, batch: Tuple[List[Union[PosixPath, str]], Tensor, Tensor]
    ):
        """Validate Generator."""
        self._val_step += 1
        fps, x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.set_grad_enabled(False):
            y_hat, _ = self._generator_step(x, y, mode="val")

        if self._val_step % self.val_output_save_interval == 0:
            self._save_val_outputs(fps, x, y, y_hat)

    def run(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        val_loader: torch.utils.data.dataloader.DataLoader,
        num_epochs: int = 1,
        max_steps: int = -1,
    ):
        t = tqdm(range(1, num_epochs + 1), leave=False, position=0)
        num_train_steps = num_epochs * len(train_loader)
        num_val_batches = len(val_loader)

        max_steps = (
            max_steps
            if (max_steps > 0 and max_steps < num_train_steps)
            else num_train_steps
        )
        max_val_batches = (
            self.max_val_batches
            if (
                self.max_val_batches > 0
                and self.max_val_batches < num_val_batches
            )
            else num_val_batches
        )
        for epoch in t:
            for train_batch in train_loader:
                self._training_step(train_batch)
                t.set_description(
                    f"Epoch: {epoch}/{num_epochs}, "
                    f"step: {self._train_step}/{max_steps}"
                )

                if self._train_step % self.val_step_interval == 0:
                    for val_batch_no, val_batch in enumerate(val_loader):
                        self._validation_step(val_batch)
                        t.set_description(
                            f"Epoch: {epoch}/{num_epochs}, "
                            f"step: {self._train_step}/{max_steps}, "
                            f"validating: {val_batch_no}/{max_val_batches}"
                        )
                        if (val_batch_no + 1) % max_val_batches == 0:
                            break
                    self._save_modules_if_min_val_loss(
                        self.metrics["val"]["gen_reconst_loss"][
                            -max_val_batches:
                        ]
                    )
                self._save_metrics()
                if (self._train_step + 1) > max_steps:
                    self._save_metrics(force_save=True)
                    return
