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


from typing import Optional

import torch
from torch import Tensor
from torch.nn import Sequential

# TRANSFORMATIONS


class Inv_Softplus(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x + torch.log(-torch.expm1(-x))


class Log(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x)


class Exp(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)


class Sqrt(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.square(x)


class Square(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)


# Maybe to convert these dictionaries in Enums ?

transforms = {"log":       Exp(),
              "softplus": torch.nn.Softplus(),
              "sqrt": Square(),
              None: None
              }

inv_transforms = {"log": Log(),
                  "softplus": Inv_Softplus(),
                  "sqrt": Sqrt(),
                  None: None
                  }


class ActivationFunction(torch.nn.Module):
    """Activation function used to parametrize the hyperparameters of the
    base_GP model. TO be used along with the torch.Parametrize class.
    """

    def __init__(self,
                 act_func: str,
                 threshold: Optional[bool] = False) -> None:
        """Initialization of the activation function
        This class permits to choose the activation function to use and offers
        the possibility to choose if a lower threshold has to be used.
        This class manage both the transform and its inverse transform.

        Args:
            act_func (str): String indicating the activation function to
            be used
            threshold (Optional[bool]): Boolean indicating if a lower
            threshold has to be used. Defaults to False.
        """
        super(ActivationFunction, self).__init__()

        self.transform = Sequential()
        self.inv_transform = Sequential()

        if act_func in transforms:
            if act_func not in inv_transforms:
                raise NotImplementedError(
                    f"Inverse Transform {act_func} has not been implemented")
            self.transform.append(transforms[act_func])
            if threshold:
                self.epsilon = 1e-6
                self.transform.add_module(
                    "thres", torch.nn.Threshold(self.epsilon, self.epsilon))
            self.inv_transform = Sequential(inv_transforms[act_func])
        else:
            raise NotImplementedError(
                f"Transform {act_func} has not been implemented")

    def forward(self, x: Tensor) -> Tensor:
        # Not sure if this is the correct way of handling the
        # self.transform = None case
        if self.transform is not None:
            return self.transform(x)
        return x

    def right_inverse(self, x: Tensor) -> Tensor:
        # Idem
        if self.inv_transform is not None:
            return self.inv_transform(x)
        return x
