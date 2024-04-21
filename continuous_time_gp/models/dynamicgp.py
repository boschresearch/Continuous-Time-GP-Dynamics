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

from typing import Any, Dict, List, Optional, Mapping

import numpy as np
import torch
from torch import Tensor
from configs.dynamicgp_config import DynamicGPConfig
from models.abstractgp import AbstractGP


class DynamicGP(torch.nn.Module):
    """Class containing a generic List of GP models that adds
    new functions for learning Dynamical systems.
    """
    models: torch.nn.ModuleList
    num_models: int
    cfg: DynamicGPConfig

    def __init__(self, model_list: List[AbstractGP]) -> None:

        super().__init__()

        self.num_models = len(model_list)
        self.models = torch.nn.ModuleList(model_list)

    def __call__(self,
                 t: Optional[float],
                 y: np.ndarray,
                 kwargs: Optional[Dict[str, Any]] = {}) -> np.ndarray:
        """This function evaluates f(t,y) learned by the GPs

        Args:
            t (float): Time
            y (np.ndarray): Spatial variable

        Returns:
            np.ndarray: Evaluation of f(t,y)
        """
        # This function has to be used with the scipy library, which
        # unfortunately has a numpy backend.
        y = np.reshape(y, (1, -1))
        f = np.zeros(self.num_models)

        # Choose a default value for ds.
        method = "posterior_mean"
        if "decoupled_sampling" in kwargs and kwargs["decoupled_sampling"]:
            method = "decoupled_sampling"

        for k in range(self.num_models):
            pred_func = getattr(self[k], method)

            # I don´t like this repeated conversion np -> torch -> np,
            # but for now it is not a bottleneck.
            f[k] = pred_func(torch.from_numpy(y), **kwargs).detach().numpy()
        return f

    def __getitem__(self, index: int) -> AbstractGP:
        """Get the models from the DynamicGP in a more clean way."""
        if not isinstance(index, int):
            raise ValueError("Index must be an integer")
        if (index < 0) or (index > self.num_models):
            raise ValueError(f"Index {index} out of bounds." +
                             f" Provide index between 0 to {self.num_models}.")
        return self.models[index]

    def resample_rff_weights(self) -> None:
        """Regenerate the random weights of the Random Fourier Features for
        every GP in the models List.
        """
        [self.models[i].resample_rff_weights() for i in range(self.num_models)]
        return

    def get_obs_noise(self) -> Tensor:
        """Return a tensor with all the observation noises for every dimension.

        Returns:
            Tensor: Tensor with all the noises.
        """
        obs_noise = []
        for i in range(len(self.models)):
            obs_noise.append(self.models[i].observation_noise())
        return torch.as_tensor(obs_noise, dtype=torch.float64).squeeze()

    def eval(self) -> None:
        self.training = False
        for i in range(len(self.models)):
            self.models[i].eval()
        return

    def to(self, device: torch.device = torch.device("cpu")) -> None:
        super().to(device)
        [model.to(device) for model in self.models]
        return

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        strict: bool = True) -> None:
        for key in ["cfg", "dataset"]:
            if key in state_dict:
                setattr(self, key, state_dict.pop(key))
        for k, model in enumerate(self.models):
            model_str = f"models.{k}."
            for field in ["cfg", "dataset", "component"]:
                key = model_str + field
                if key in state_dict:
                    setattr(model, field, state_dict.pop(key))
        super().load_state_dict(state_dict, strict)
        return

    def state_dict(self,
                   destination=None,
                   prefix='',
                   keep_vars=False) -> Mapping[str, Any]:
        """ Overrides state_dict() to save also cfg and dataset"""
        original_dict = super().state_dict(destination=destination,
                                           prefix=prefix,
                                           keep_vars=keep_vars)
        original_dict[prefix+'cfg'] = self.cfg
        original_dict[prefix+'dataset'] = self.models[0].dataset
        return original_dict
