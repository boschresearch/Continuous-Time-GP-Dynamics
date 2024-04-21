# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Nicholas Tagliapietra, Katharina Ensinger (katharina.ensinger@bosch.com)

import os
import sys
from typing import Any, Callable, Mapping, Optional
import logging

import torch.multiprocessing as mp
import torch
from configs.train_config import TrainConfig
from models.abstractgp import AbstractGP
from models.dynamicgp import DynamicGP
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from utils.losses import marginalLogLikelihoodLoss
from utils.train_utils import load_abstract_gp
from copy import deepcopy
from configs.gp_config import MultistepGPConfig, TaylorGPConfig, ExactTaylorGPConfig


def train_dynamicGP(gp: DynamicGP,
                    train_cfg: TrainConfig,
                    exp_dir: str,
                    device: torch.device
                    ) -> DynamicGP:
    """Train a DynamicGP with the configuration in train_cfg, on a specific
    device, and store everything in exp_dir.

    Args:
        gp (Dynamic_GP): Dynamic_GP containing a list of GP models.
        train_config (TrainConfig): Training config
        exp_dir (str): Experiment folder.
        device (torch.device): Device to train the models on.

    Raises:
        TypeError: The model does not subclass AbstractGP

    Returns:
        Dynamic_GP: Trained dynamic_GP model.
    """
    for model in gp.models:
        assert isinstance(model,
                          AbstractGP), "Model is not an AbstractGP child"

    num_models = len(gp.models)

    loss_func = marginalLogLikelihoodLoss
    epochs = train_cfg.epochs
    train_dir = os.path.join(exp_dir, "training")

    args = [(gp.models[i].state_dict(),
             loss_func,
             train_cfg,
             train_dir,
             device) for i in range(num_models)]

    torch.set_num_threads(1)

    if isinstance(gp.models[0].cfg, ExactTaylorGPConfig):
        gp_models = train_exact_taylor(gp.models, loss_func, train_cfg, train_dir, device)

    else:
        with mp.Pool() as pool:
            pool.starmap(train_model, args)
    for model in gp.models:
        model_str = f"model_{model.component}"
        filename = f"{str(epochs - 1)}.pt"
        checkpoint = torch.load(os.path.join(train_dir,
                                             model_str,
                                             filename))
        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict)

    return gp


def set_tensorboard(exp_path: str) -> SummaryWriter:
    """Given a raw path, set the SummaryWriter and the path. Data will be saved
    the folder ../tensorboard_data/<DateTime>_<ExperimentName>_<Seed>.

    Args:
        exp_path (str): Path to the experiment folder to be parsed.

    Returns:
        SummaryWriter: Tensorboard callback
    """
    s = "_"  # Separator
    tensorboard_dir = os.path.join(sys.path[0],
                                   "tensorboard_data",
                                   s.join(exp_path.split('\\')[-2:]))
    writer = SummaryWriter(tensorboard_dir)
    return writer


def train_model(state_dict: Optional[Mapping[str, Any]],
                likelihood_func: Callable,
                train_cfg: TrainConfig,
                train_dir: str,
                device: torch.device) -> None:
    """Training function for a model inheriting from an AbstractGP model.

    Args:
        state_dict (Optional[Mapping[str, Any]]): State dictionary for a GP model
        likelihood_func (Callable): Likelihood function.
        train_cfg (TrainConfig): Config object for training.
        train_dir (str): Path where to save the intermediate results.
        device (torch.device): Device to train
    """
    print_onscreen = train_cfg.print_onscreen

    model = load_abstract_gp(state_dict)
    model_str = f"model_{model.component}"

    model.to(device)
    dataloader = model.dataset.get_dataloader(model.component)

    tb_writer = set_tensorboard(train_dir)

    root_path = os.path.join(train_dir,
                             model_str)

    n_epochs = train_cfg.epochs
    learning_rate = train_cfg.lr
    optimizer_func = getattr(torch.optim, train_cfg.algorithm)

    # Model in training mode
    model.train()

    # Define optimizer
    optimizer = optimizer_func(model.parameters(), lr=learning_rate)

    loss_history = []

    for epoch in range(n_epochs):
        for inputs, targets in dataloader:
            try:
                optimizer.zero_grad()

                output = model.forward(inputs)
                loss = - likelihood_func(output, targets)

                loss.backward(retain_graph = False)

                loss_history.append(loss.detach().item())
                # Save data
                if ((epoch) % 50 == 0):
                    tb_writer.add_scalar(f'Train/Loss/{model_str}',
                                         loss,
                                         epoch)
                    model.log_tensorboard(tb_writer,
                                          epoch,
                                          model_str)

                    if root_path is not None and ((epoch) % 500 == 0):
                        save_results(model,
                                     root_path,
                                     optimizer,
                                     epoch,
                                     loss)

                    if print_onscreen:
                        print(f"Iter {epoch:5}/{n_epochs - 1} - Loss: ",
                              f"{loss.item():10}", end="\r")

                optimizer.step()

            except KeyboardInterrupt:
                logging.info("Training Stopped")
                break

    logging.info("Training ended")

    torch.save(loss_history, os.path.join(root_path, "loss.pt"))

    # Just one batch without backward pass,
    # used to evaluate the final loss and save the final model
    if root_path is not None:
        for inputs, targets in dataloader:
            output = model.forward(inputs)
            loss = - likelihood_func(output, targets)
            save_results(model,
                         root_path,
                         optimizer,
                         n_epochs - 1,
                         loss)

    if print_onscreen:
        print("")
    tb_writer.flush()
    return

def train_exact_taylor(gp_models,
                likelihood_func: Callable,
                train_cfg: TrainConfig,
                train_dir: str,
                device: torch.device) -> None:

    Args:
        state_dict (Optional[Mapping[str, Any]]): State dictionary for a GP model
        likelihood_func (Callable): Likelihood function.
        train_cfg (TrainConfig): Config object for training.
        train_dir (str): Path where to save the intermediate results.
        device (torch.device): Device to train
    """
    print_onscreen = train_cfg.print_onscreen
    dataset = deepcopy(gp_models[0].dataset)
    gp_models.to(device)
    dataloader = dataset.get_full_dataloader()

    tb_writer = set_tensorboard(train_dir)

    root_path = [os.path.join(train_dir,f"model_{i}") for i in range(len(gp_models))]

    n_epochs = train_cfg.epochs
    learning_rate = train_cfg.lr
    optimizer_func = getattr(torch.optim, train_cfg.algorithm)

    # Model in training mode
    gp_models.train()

    # Define optimizer
    optimizer = optimizer_func(gp_models.parameters(), lr=learning_rate)

    loss_history = []

    for epoch in range(n_epochs):
        for inputs, targets in dataloader:
            try:
                optimizer.zero_grad()
                outputs = [gp.forward(inputs) for gp in gp_models]
                losses = [likelihood_func(outputs[i], targets[:, i]) for i in range(len(outputs))]
                loss = - torch.stack(losses).sum()
                loss.backward(retain_graph = False)

                loss_history.append(loss.detach().item())
                # Save data
                if ((epoch) % 50 == 0):
                    for i in range(len(gp_models)):
                        model_str = f"model_{i}"
                        tb_writer.add_scalar(f'Train/Loss/{model_str}',
                                             loss,
                                             epoch)
                        gp_models[i].log_tensorboard(tb_writer,
                                              epoch,
                                              model_str)

                        if root_path is not None and ((epoch) % 500 == 0):
                            save_results(gp_models[i],
                                         root_path[i],
                                         optimizer,
                                         epoch,
                                         loss)

                        if print_onscreen:
                            print(f"Iter {epoch:5}/{n_epochs - 1} - Loss: ",
                                  f"{loss.item():10}", end="\r")

                optimizer.step()

            except KeyboardInterrupt:
                logging.info("Training Stopped")
                break

    logging.info("Training ended")

    torch.save(loss_history, os.path.join(root_path[0], "loss.pt"))

    # Just one batch without backward pass,
    # used to evaluate the final loss and save the final model
    if root_path is not None:
        for inputs, targets in dataloader:
            outputs = [gp.forward(inputs) for gp in gp_models]
            losses = [likelihood_func(outputs[i], targets[:, i]) for i in
                      range(len(outputs))]
            loss = - torch.stack(losses).sum()
            for i in range(len(gp_models)):
                save_results(gp_models[i],
                             root_path[i],
                             optimizer,
                             epoch,
                             loss)
    if print_onscreen:
        print("")
    tb_writer.flush()
    return gp_models


def save_results(model: AbstractGP,
                 root_path: str,
                 optimizer,
                 epoch: int,
                 loss: Tensor) -> None:
    """Save the intermediate results during training.

    Args:
        model (AbstractGP): Model during training.
        root_path (str): Path to save results.
        optimizer (_type_): Optimizer.
        epoch (int): Epoch number.
        loss (Tensor): Loss value at epoch.
    """
    if root_path is not None:
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        torch.save(
            {"epoch": epoch,
             "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "loss": loss,
             },
            root_path + f"/{epoch}.pt")
