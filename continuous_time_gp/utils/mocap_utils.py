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

#from src.misc.settings import settings

from typing import List, Tuple

import numpy as np
import torch
from data.datasets.mocap import MocapDataset
from torch import Tensor


class Latent2DataProjector:
    """
    Defines latent/PCA space to observation space transformations
    """

    def __init__(self, dataset: MocapDataset):
        self.data_std = torch.tensor(dataset.data_std.astype(np.float64))
        self.data_mean = torch.tensor(dataset.data_mean.astype(np.float64))

        if dataset.pca_normalize is not None:
            self.pca_normalize_mean = torch.tensor(
                dataset.pca_normalize.mean.astype(np.float64))
            self.pca_normalize_std = torch.tensor(
                dataset.pca_normalize.std.astype(np.float64))
            self.inverse_pca_normalization = lambda x: (
                x * self.pca_normalize_std) + self.pca_normalize_mean
        else:
            self.inverse_pca_normalization = lambda x: x

        self.pca_components = torch.tensor(
            dataset.pca.components_.astype(np.float64))
        self.inverse_pca = lambda x: torch.einsum(
            'ntl,ld->ntd', x, self.pca_components)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.inverse_pca_normalization(x)
        x = self.inverse_pca(x)
        return x.squeeze()


def mocap_project_list(traj_list: List[Tuple[Tensor, Tensor]],
                       projector
                       ) -> List[Tuple[Tensor, Tensor]]:
    """Convert a list of trajectories in the latent space to a list of
    trajectories in the original space.

    Args:
        projector (_type_): Projector from latent2data space
        traj_list (List[Tuple[Tensor, Tensor]]): List of trajectories in latent
        PCA space

    Returns:
        List[Tuple[Tensor, Tensor]]: List of trajectories in the original space
    """
    new_traj_list = []
    for i in range(len(traj_list)):
        new_traj_list.append((projector(traj_list[i][0]), traj_list[i][1]))
    return new_traj_list
