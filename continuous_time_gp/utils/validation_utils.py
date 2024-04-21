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
from typing import List, Tuple

import torch
from data.datasets.mocap import MocapDataset
from torch import Tensor
from utils.mocap_utils import Latent2DataProjector
from configs.data_config import MocapConfig


def build_mocap_test(dataset_cfg: MocapConfig
                     ) -> Tuple[Latent2DataProjector,
                                List[Tuple[Tensor, Tensor]]]:
    """Given a config for mocap, return the projector from PCA to original
    space and the list of test trajectories in the right format for this
    framework.

    Args:
        dataset_cfg (DatasetConfig): Config for mocap

    Returns:
        Tuple[Latent2DataProjector, List[Tuple[Tensor, Tensor]]]: Projector and 
        list of test trajectories.
    """
    if not isinstance(dataset_cfg, MocapConfig):
        raise TypeError("dataset_cfg is not a MocapConfig")

    data_path = os.path.join(sys.path[0], "data", "raw", "mocap")

    data_pca = MocapDataset(data_path=data_path,
                            subject=dataset_cfg.subject_num,
                            pca_components=dataset_cfg.pca_components,
                            data_normalize=dataset_cfg.data_normalize,
                            pca_normalize=dataset_cfg.pca_normalize,
                            dt=dataset_cfg.dt,
                            seqlen=dataset_cfg.seqlen)

    projector = Latent2DataProjector(data_pca)

    data_full = MocapDataset(data_path=data_path,
                             subject=dataset_cfg.subject_num,
                             pca_components=-1,
                             data_normalize=False,
                             pca_normalize=False,
                             dt=dataset_cfg.dt,
                             seqlen=dataset_cfg.seqlen)

    test_ys = torch.tensor(data_full.tst.ys)
    test_ts = torch.tensor(data_full.tst.ts)
    test_ys_pca = torch.tensor(data_pca.tst.ys)
    test_ts_pca = torch.tensor(data_pca.tst.ts)

    # EVALUATE DISTANCE BETWEEN PCA TRANSFORMED DATA AND ORIGINAL DATA

    # test_ys_pca = projector(torch.tensor(data_pca.tst.ys).double())

    # mse_list = []
    # for i in range(test_ys_pca.shape[0]):

    #     err = torch.square(test_ys[i, :, :] - test_ys_pca[i, :, :])
    #     mse = torch.mean(err)
    #     mse_list.append(mse)

    ###

    ntraj = test_ys.shape[0]
    mocap_eval = [(test_ys[i, :, :], test_ts) for i in range(ntraj)]
    mocap_eval_pca = [(test_ys_pca[i, :, :], test_ts_pca)
                      for i in range(ntraj)]

    return projector, mocap_eval, mocap_eval_pca
