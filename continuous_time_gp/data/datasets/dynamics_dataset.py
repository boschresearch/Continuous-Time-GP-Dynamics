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

import copy
from typing import List, Optional, Tuple

import torch
from configs.data_config import MethodEnum
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from train.create_training_data import (create_training_data_multistep,
                                        create_training_data_taylor)
from utils.numerical_schemes import NumericalScheme


class DynamicsDataset(Dataset):
    """Class used for managing Datasets containing trajectories,
    to be used within the framework of multistep_GPs and Taylor_GPs.
    """
    data: Tuple[Tensor]
    targets: Tensor
    method: MethodEnum
    timeline: Tensor
    scheme: Optional[NumericalScheme]

    def __init__(self,
                 train_list: List[Tuple[Tensor, Tensor]],
                 method: MethodEnum,
                 scheme: Optional[NumericalScheme] = None) -> None:
        """Initialize a DynamicsDataset.

        Args:
            trajectory_tuple (Tuple[Tensor, Tensor]): Tuple containing the
            trajectory data X along with the correspondent timeline.
            method (MethodEnum): Indicates the method to use.
            scheme (Optional[NumericalScheme], optional): For "multistep"
            case, a function to generate the coefficients has to be provided.
            Defaults to None.

        Raises:
            ValueError: Raised when the method is unknown.
        """
        super().__init__()

        self.method = method

        if self.method not in MethodEnum:
            raise TypeError("Unknown method!")

        if self.method == MethodEnum.Multistep:
            if not isinstance(scheme(), NumericalScheme):
                raise TypeError("scheme is not a NumericalScheme")
            else:
                self.scheme = scheme

        self.timelines = []
        for i in range(len(train_list)):
            traj = train_list[i][0]
            timeline = train_list[i][1]
            self.timelines.append(timeline)
            
            match self.method:
                case "taylor":
                    x, y, _, epsilon = create_training_data_taylor(
                        traj, timeline)
                    self.concat_data(x, y, epsilon)
                case "multistep":
                    x, y, _, epsilon = create_training_data_multistep(
                        traj, timeline, scheme)

                    (alpha, beta) = self.scheme.build_coeffs(timeline)

                    self.concat_data(x, y, epsilon, alpha=alpha, beta=beta)

        if not all(x[0].shape == tensor.shape for tensor in x):
            raise RuntimeError("Shape mismatch between training tensors")
        if not x[0].shape == y.shape:
            raise RuntimeError("Shape mismatch between inputs and targets")

        return

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor]:
        x = [tensor[index] for tensor in self.data]
        y = self.targets[index]
        return (x, y)

    def __len__(self) -> int:
        return self.data[0].shape[0]

    def to(self, device: torch.device) -> None:
        self.data = tuple([self.data[i].to(device)
                          for i in range(len(self.data))])
        self.targets = self.targets.to(device)
        return

    def get_full_dataloader(self) -> DataLoader:
        data = copy.deepcopy(self)
        data.targets = data.targets[:]
        return DataLoader(data, batch_size = len(self))

    def get_dataloader(self, component: int) -> DataLoader:
        """Get a dataloader ready for training a GP on a specific component
        of the target data.

        Args:
            component (int): Component to extract from the targets.

        Returns:
            DataLoader: Dataloader ready for training an exact GP
            (Batch equal to full dataset).
        """
        data = copy.deepcopy(self)
        data.targets = data.targets[:, component]
        return DataLoader(data, batch_size=len(self))

    def concat_data(self,
                    x: List[Tuple[Tensor, Tensor]],
                    y: Tensor,
                    epsilon: Tensor,
                    alpha: Optional[Tensor] = None,
                    beta: Optional[Tensor] = None) -> None:
        """In case of multiple training trajectories, concatenate the new 
        training data to existing data. This function takes care also of the
        initializations of the data tensors, in case it is the first
        trajectory.
        """

        if not hasattr(self, "data"):
            # If this is the first trajectory, store the input without
            # concatenation
            # Maybe an exception to the function signature, but...
            self.data = list(x)
            self.targets = y
            self.epsilon = epsilon
        else:
            # Concatenate the new datapoints
            for j in range(len(x)):
                self.data[j] = torch.cat((self.data[j], x[j]),
                                         dim=0)
            self.targets = torch.cat((self.targets, y),
                                     dim=0)
                                     
            # OLD CODE
            # self.epsilon = torch.cat((self.epsilon, epsilon),
            #                         dim=0)

            # NEW
            # We consider each trajectory independent, so the noise matrix will 
            # result having a block-diagonal structure
            self.epsilon = torch.block_diag(self.epsilon, epsilon)

        if alpha is not None:
            if not hasattr(self, "alpha"):
                # If this is the first trajectory, store the input without
                # concatenation
                self.alpha = alpha
                self.beta = beta
                return

            self.alpha = torch.cat((self.alpha, alpha),
                                   dim=0)
            self.beta = torch.cat((self.beta, beta),
                                  dim=0)

        return

    @property
    def data_dimension(self) -> int:
        return self.targets.shape[1]
