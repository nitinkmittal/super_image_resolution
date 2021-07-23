import logging
import multiprocessing as mp
import os
from collections import OrderedDict
from pathlib import PosixPath
from time import time
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)
from tqdm import tqdm

from super_image_resolution import get_root_path
from super_image_resolution.datasets import CustomDataset
from super_image_resolution.models.unet import UNet
from super_image_resolution.transforms import CustomResize
from super_image_resolution.utils import (
    compute2d_means_and_stds,
    dump,
    find_latest_model_version,
    load,
    make_dirs,
    save_output,
    split,
)

HPARAMS = OrderedDict()

HPARAMS["VERBOSE"] = True
HPARAMS["PROJECT_PATH"] = get_root_path()
HPARAMS["ASSETS_PATH"] = os.path.join(HPARAMS["PROJECT_PATH"], "assets")
HPARAMS["DATA_PATH"] = os.path.join("/", "scratch", "mittal.nit", "faces-spring-2020")

HPARAMS["MODEL_NAME"] = "unet_l1loss_batchnorm48to128"
HPARAMS["MODEL_PATH"] = os.path.join(HPARAMS["ASSETS_PATH"], HPARAMS["MODEL_NAME"])
make_dirs(HPARAMS["ASSETS_PATH"])
make_dirs(HPARAMS["MODEL_PATH"])


HPARAMS["WEIGHTS_PATH"] = os.path.join(HPARAMS["MODEL_PATH"], "weights")
HPARAMS["MODEL_WEIGHTS"] = "model"
HPARAMS["OPTIM_WEIGHTS"] = "optim"
HPARAMS["LR_SCH_WEIGHTS"] = "lr_sch"

HPARAMS["VERSION"] = find_latest_model_version(HPARAMS["MODEL_PATH"]) + 1
if HPARAMS["VERBOSE"]:
    print(f"Current version: {HPARAMS['VERSION']}")
# HPARAMS["VERSION"] = 0
HPARAMS["EXPERIMENT_VERSION_PATH"] = os.path.join(
    HPARAMS["MODEL_PATH"], f"version_{HPARAMS['VERSION']}"
)
HPARAMS["LOG"] = "log"
HPARAMS["HPARAMS"] = "hparams"
HPARAMS["METRICS"] = "metrics"
HPARAMS["IMAGES_PATH"] = os.path.join(
    HPARAMS["EXPERIMENT_VERSION_PATH"], "images", "val"
)


make_dirs(HPARAMS["WEIGHTS_PATH"])
make_dirs(HPARAMS["EXPERIMENT_VERSION_PATH"])
make_dirs(HPARAMS["IMAGES_PATH"])

HPARAMS["SPLIT_RATIOS"] = (0.85, 0.10, 0.05)
HPARAMS["RANDOM_SEED"] = 40

HPARAMS["TRAIN_BATCH_SIZE"] = 64
HPARAMS["VAL_BATCH_SIZE"] = 128
HPARAMS["TEST_BATCH_SIZE"] = 128
HPARAMS["NUM_WORKERS"] = max(0, (mp.cpu_count() - 7) // 3)

HPARAMS["PRE_TRAINED"] = False
HPARAMS["IN_CHANNELS"] = 3
HPARAMS["OUT_CHANNELS"] = 3
HPARAMS["ACTIVATION"] = "lrelu"
HPARAMS["FINAL_ACTIVATION"] = "tanh"
HPARAMS["NORM"] = "batchnorm"
HPARAMS["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
# HPARAMS["DEVICE"] = "cpu"

# HPARAMS["DATA_IN_NORM"] = {
#     "means": [0.5351, 0.4395, 0.3800],
#     "stds": [0.2339, 0.2071, 0.2065],
# }

# HPARAMS["DATA_OUT_NORM"] = {
#     "means": [0.5345, 0.4390, 0.3796],
#     "stds": [0.2353, 0.2083, 0.2074],
# }
HPARAMS["SCALE_Y"] = 1.0

HPARAMS["EPOCHS"] = 500
HPARAMS["LR"] = 5e-4
HPARAMS["BETAS"] = (0.5, 0.9)

HPARAMS["LR_STEP_SIZE"] = 100
HPARAMS["LR_GAMMA"] = 0.999

HPARAMS["INBETWEEN_SAVE_METRICS_INTERVAL"] = 25
HPARAMS["MAX_VAL_SAVE_SAMPLES"] = min(5, HPARAMS["VAL_BATCH_SIZE"])
HPARAMS["VAL_RANDOM_SAVE_SAMPLES"] = True

if HPARAMS["VERBOSE"]:
    print("Defininig transformations...")
in_transforms = Compose(
    [
        CustomResize(size=48, interpolation=InterpolationMode.BICUBIC),
        CustomResize(size=128, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
    ]
)
out_transforms = Compose(
    [
        Resize(size=128, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
    ]
)

if HPARAMS["VERBOSE"]:
    print("Splitting data into train, val and test set...")
train_path, val_path, test_path = split(
    path=HPARAMS["DATA_PATH"],
    ratios=HPARAMS["SPLIT_RATIOS"],
    seed=HPARAMS["RANDOM_SEED"],
)

HPARAMS["NUM_TRAIN_SAMPLES"] = len(train_path)
HPARAMS["NUM_VAL_SAMPLES"] = len(val_path)
HPARAMS["NUM_TEST_SAMPLES"] = len(test_path)

if HPARAMS["VERBOSE"]:
    print("Initializing datasets...")
train_dataset = CustomDataset(
    paths=train_path,
    in_transforms=in_transforms,
    out_transforms=out_transforms,
)
val_dataset = CustomDataset(
    paths=val_path, in_transforms=in_transforms, out_transforms=out_transforms
)
test_dataset = CustomDataset(
    paths=test_path, in_transforms=in_transforms, out_transforms=out_transforms
)

if HPARAMS["VERBOSE"]:
    print("Initializing dataloaders...")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=HPARAMS["TRAIN_BATCH_SIZE"],
    shuffle=True,
    num_workers=HPARAMS["NUM_WORKERS"],
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=HPARAMS["VAL_BATCH_SIZE"],
    shuffle=False,
    num_workers=HPARAMS["NUM_WORKERS"],
)
test_loader = DataLoader(
    dataset=val_dataset,
    batch_size=HPARAMS["TEST_BATCH_SIZE"],
    shuffle=False,
    num_workers=HPARAMS["NUM_WORKERS"],
)

HPARAMS["NUM_TRAIN_BATCHES"] = len(train_loader)
HPARAMS["NUM_VAL_BATCHES"] = len(val_loader)
HPARAMS["NUM_TEST_BATCHES"] = len(test_loader)
HPARAMS["INBETWEEN_TRAIN_VAL_CHECK_INTERVAL"] = min(500, HPARAMS["NUM_TRAIN_BATCHES"])


if HPARAMS["VERBOSE"]:
    print("Initializing model...")
model = UNet(
    in_channels=HPARAMS["IN_CHANNELS"],
    out_channels=HPARAMS["OUT_CHANNELS"],
    norm=HPARAMS["NORM"],
    act=HPARAMS["ACTIVATION"],
)
model.to(HPARAMS["DEVICE"])

# Loading pre-trained model, if exists
if HPARAMS["PRE_TRAINED"] and HPARAMS["MODEL_WEIGHTS"] in os.listdir(
    HPARAMS["WEIGHTS_PATH"]
):
    try:
        model.load_state_dict(
            torch.load(os.path.join(HPARAMS["WEIGHTS_PATH"], HPARAMS["MODEL_WEIGHTS"]))
        )
        if HPARAMS["VERBOSE"]:
            print("Loaded pre-trained model weights")
    except Exception as e:
        print(f"Error: {e}")

criterion = nn.L1Loss()
optim = torch.optim.Adam(model.parameters(), lr=HPARAMS["LR"], betas=HPARAMS["BETAS"])
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optim,
    step_size=HPARAMS["LR_STEP_SIZE"],
    gamma=HPARAMS["LR_GAMMA"],
)

# Loading pre-trained optimizer, if exists
if HPARAMS["PRE_TRAINED"] and HPARAMS["OPTIM_WEIGHTS"] in os.listdir(
    HPARAMS["WEIGHTS_PATH"]
):
    try:
        optim.load_state_dict(
            torch.load(os.path.join(HPARAMS["WEIGHTS_PATH"], HPARAMS["OPTIM_WEIGHTS"]))
        )
        logging.info("Loaded pre-trained optimizer")
    except Exception as e:
        logging.info(f"Error: {e}")

# Loading pre-trained lr scheduler, if exists
if HPARAMS["PRE_TRAINED"] and HPARAMS["LR_SCH_WEIGHTS"] in os.listdir(
    HPARAMS["WEIGHTS_PATH"]
):
    try:
        lr_scheduler.load_state_dict(
            torch.load(os.path.join(HPARAMS["WEIGHTS_PATH"], HPARAMS["LR_SCH_WEIGHTS"]))
        )
        if HPARAMS["VERBOSE"]:
            print("Loaded pre-trained lr_scheduler")
    except Exception as e:
        print(f"Error: {e}")

# Setting metrics
metrics = OrderedDict()
metrics["train"] = OrderedDict(step=[], epoch=[])
metrics["val"] = OrderedDict(step=[], epoch=[])
metrics["min_val_loss"] = 1e10


def training_step(model, X: Tensor, Y: Tensor) -> float:
    """Training step for Pytorch model."""
    torch.set_grad_enabled(True)
    model.train()
    Y_hat = HPARAMS["SCALE_Y"] * model(X)
    loss = criterion(Y, Y_hat)
    optim.zero_grad()
    loss.backward()
    optim.step()
    lr_scheduler.step()
    return loss.item()


def validation_step(model, X: Tensor, Y: Tensor) -> Tuple[float, Tensor]:
    """Validation step for Pytorch model."""
    torch.set_grad_enabled(False)
    model.eval()
    Y_hat = HPARAMS["SCALE_Y"] * model(X)
    loss = criterion(Y, Y_hat)
    metrics["val"]["step"].append(loss.item())
    return loss.item(), Y_hat


if HPARAMS["VERBOSE"]:
    print("Saving experiment configuration...")
dump(
    os.path.join(HPARAMS["EXPERIMENT_VERSION_PATH"], HPARAMS["HPARAMS"]),
    HPARAMS,
)

train_step_no, val_step_no = 0, 0
t = tqdm(range(HPARAMS["EPOCHS"]), leave=False)
for epoch_i in t:
    ###################### train mode ######################
    epoch_loss = 0.0
    n = 0
    for (X, Y) in train_loader:
        X, Y = X.to(HPARAMS["DEVICE"]), Y.to(HPARAMS["DEVICE"])

        loss = training_step(model, X, Y)
        metrics["train"]["step"].append(loss)
        info = f"mode: train, epoch: {epoch_i}, step: {train_step_no}, step_loss: {loss: .6f}"
        t.set_description(info)
        logging.info(info)

        epoch_loss += loss * len(X)
        n += len(X)
        train_step_no += 1

        # intermediate val outputs
        if train_step_no % HPARAMS["INBETWEEN_TRAIN_VAL_CHECK_INTERVAL"] == 0:
            for val_batch_i, (X, Y) in enumerate(val_loader):
                if 0.5 > np.random.uniform():  # not save all intermediate results
                    X, Y = X.to(HPARAMS["DEVICE"]), Y.to(HPARAMS["DEVICE"])
                    loss, Y_hat = validation_step(model, X, Y)
                    save_output(
                        in_=X,
                        out_true=Y,
                        out_pred=Y_hat,
                        fp=os.path.join(
                            HPARAMS["IMAGES_PATH"],
                            f"train_step_{train_step_no}_val_batch_{val_batch_i}",
                        ),
                        max_save=HPARAMS["MAX_VAL_SAVE_SAMPLES"],
                        random_save=HPARAMS["VAL_RANDOM_SAVE_SAMPLES"],
                    )

        # save metrics
        if train_step_no % HPARAMS["INBETWEEN_SAVE_METRICS_INTERVAL"] == 0:
            dump(
                os.path.join(HPARAMS["EXPERIMENT_VERSION_PATH"], HPARAMS["METRICS"]),
                metrics,
            )

    epoch_loss /= n
    metrics["train"]["epoch"].append(epoch_loss)

    info = f"mode: train, epoch: {epoch_i}, epoch_loss: {epoch_loss: .6f}"
    t.set_description(info)
    logging.info(info)

    ###################### val mode ######################
    epoch_loss = 0.0
    n = 0
    for X, Y in val_loader:
        X, Y = X.to(HPARAMS["DEVICE"]), Y.to(HPARAMS["DEVICE"])
        loss, _ = validation_step(model, X, Y)
        metrics["val"]["step"].append(loss)
        info = (
            f"mode: val, epoch: {epoch_i}, step: {val_step_no}, step_loss: {loss: .6f}"
        )
        t.set_description(info)
        logging.info(info)

        epoch_loss += loss * len(X)
        n += len(X)
        val_step_no += 1

    epoch_loss /= n
    metrics["val"]["epoch"].append(epoch_loss)

    info = f"mode: val, epoch: {epoch_i}, epoch_loss: {epoch_loss: .6f}"
    t.set_description(info)
    logging.info(info)

    # saving best model weights on basis of val epoch metrics
    if epoch_loss <= metrics["min_val_loss"]:
        if HPARAMS["VERBOSE"]:
            print("Best model found, saving model, optimizer and lr_scheduler weights")
        metrics["min_val_loss"] = epoch_loss
        torch.save(
            model.state_dict(),
            os.path.join(HPARAMS["WEIGHTS_PATH"], HPARAMS["MODEL_WEIGHTS"]),
        )
        torch.save(
            optim.state_dict(),
            os.path.join(HPARAMS["WEIGHTS_PATH"], HPARAMS["OPTIM_WEIGHTS"]),
        )
        torch.save(
            lr_scheduler.state_dict(),
            os.path.join(HPARAMS["WEIGHTS_PATH"], HPARAMS["LR_SCH_WEIGHTS"]),
        )

dump(
    os.path.join(HPARAMS["EXPERIMENT_VERSION_PATH"], HPARAMS["METRICS"]),
    metrics,
)
