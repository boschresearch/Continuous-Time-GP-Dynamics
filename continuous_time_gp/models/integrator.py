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

import math
from abc import ABC, abstractclassmethod
from copy import deepcopy
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch
from data.dynamical_models import ode_solution
from factories.dynamicgp_factory import DynamicGPFactory
from models.dynamicgp import DynamicGP
from models.taylorgp import TaylorGP
from scipy.optimize import least_squares
from torch import Tensor
from utils.numerical_schemes import NumericalScheme

class AbstractIntegrator(torch.nn.Module, ABC):
    """Base class for every integrator."""

    steps: int
    scheme: NumericalScheme
    name: str

    def __init__(self) -> None:
        super().__init__()
        return

    @abstractclassmethod
    def integrate_n_steps(self,
                          y: Tensor,
                          timeline: Tensor,
                          f: Optional[DynamicGP] = None,
                          model_state_dict: Optional[Mapping[str, Any]] = None,
                          int_state_dict: Optional[Mapping[str, Any]] = None,
                          **kwargs) -> Tensor:
        """Integrate the dynamics described by f along timeline
        with initial conditions y.

        Args:
            y (Tensor): Initial steps needed for starting
            the integration.
            timeline (Tensor): Timeline over which to integrate.
            f (Dynamic_GP): Function describing the dynamics.
            model_state_dict (Optional[Mapping[str, Any]]): State dictionary of 
            the model.
            int_state_dict (Optional[Mapping[str, Any]]): State dictionary of 
            the integrator.

        Returns:
            Tensor: Approximated dynamics of the system.
        """
        raise NotImplementedError

    def init_model(self,
                   state_dict: Optional[Mapping[str, Any]] = None,
                   f: Optional[DynamicGP] = None) -> DynamicGP:
        """Initialize GP from state_dict

        Args:
            state_dict (Optional[Mapping[str, Any]], optional): _description_.
            Defaults to None.
            f (Optional[DynamicGP], optional): _description_. Defaults to None.

        Raises:
            RuntimeError: _description_

        Returns:
            DynamicGP: _description_
        """
        if state_dict is not None:
            cfg = deepcopy(state_dict["cfg"])
            dataset = deepcopy(state_dict["dataset"])
            dataset.to("cpu")

            # Initialize GP and load state dict
            f = DynamicGPFactory.build(cfg, dataset)
            f.load_state_dict(state_dict)
            f.to("cpu")
            f.eval()
        elif f is None:
            raise RuntimeError("Model or State_dict not provided.")

        return f

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        strict: bool = True) -> None:
        if hasattr(self, "scheme"):
            setattr(self.scheme, "k", state_dict.pop("scheme.k"))
            setattr(self.scheme, "name", state_dict.pop("scheme.name"))
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
        if hasattr(self, "scheme"):
            original_dict[prefix+'scheme.k'] = self.scheme.k
            original_dict[prefix+'scheme.name'] = self.scheme.name
        return original_dict


class MultistepIntegrator():
    @staticmethod
    def build(scheme: NumericalScheme) -> AbstractIntegrator:
        """Factory method with the task of generating the correct integrator
        starting from the coefficient generator.

        Args:
            scheme (NumericalScheme): Functions containing the coefficient
            generator for a specific multistep method.

        Returns:
            AbstractIntegrator: An implicit or explicit integrator.
        """
        # Generate a matrix of coefficients starting from a simple and short
        # timeline, then extract nsteps and num

        nsteps = scheme.get_k()

        # Maybe add an "implicit" attribute
        _, beta = scheme.build_coeffs(torch.linspace(0, 1, 10))
        # If the k-th beta coefficient is almost zero, the method is implicit.
        implicit = torch.any(beta[:, -1] > 1e-6).item()

        integrator = ImplicitIntegrator() if implicit else ExplicitIntegrator()

        integrator.scheme = scheme
        integrator.steps = torch.tensor(nsteps)
        return integrator


class ImplicitIntegrator(AbstractIntegrator):

    def __init__(self) -> None:
        super().__init__()
        return

    def _integration_step(self,
                          f: DynamicGP,
                          y: np.ndarray,
                          h: np.ndarray,
                          alpha: np.ndarray,
                          beta: np.ndarray,
                          **kwargs) -> np.ndarray:
        """Perform an integration step for an implicit numerical scheme
        described by the coefficients alpha and beta.

        Args:
            f (Dynamic_GP): Function describing the dynamics f(t,y)
            y (np.ndarray): Past steps.
            h (np.ndarray): Stepsize at time t_{n + k - 1}
            alpha (np.ndarray): Past alpha coefficients
            beta (np.ndarray): Past beta coefficients

        Returns:
            np.ndarray: Next step y_{n+k}
        """

        def implicit_system(ynext, y_last, f, alpha, beta, h, kwargs):
            # The implicit system to be solved during the integration phase
            alpha_t = alpha.T
            beta_t = beta.T
            explicit_term = np.sum(np.multiply(y_last, alpha_t), axis=0)

            implicit_term = np.zeros((y_last.shape[1],))
            for i in range(self.steps):
                implicit_term = implicit_term + \
                    np.multiply(beta_t[i], f(None, y_last[i, :], kwargs))
            implicit_term = implicit_term + \
                np.multiply(beta[:, -1], f(None, ynext, kwargs))
            res = ynext + explicit_term - h * implicit_term
            return res

        result = least_squares(implicit_system,
                               y[self.steps - 1, :],
                               args=(y, f, alpha, beta, h, kwargs),
                               method='lm')
        # .x because it returns an object OptimizerResult

        if result.success is False:
            raise RuntimeError("Least Squares failed to converge.")

        return result.x

    def integrate_n_steps(self,
                          y: Tensor,
                          timeline: Tensor,
                          f: Optional[DynamicGP] = None,
                          model_state_dict: Optional[Mapping[str, Any]] = None,
                          int_state_dict: Optional[Mapping[str, Any]] = None,
                          **kwargs) -> Tensor:
        torch.set_num_threads(1)

        f = self.init_model(state_dict=model_state_dict, f=f)
        self.load_state_dict(int_state_dict)

        # Necessary to handle the am_0 case where nsteps == 0
        if self.scheme.name in ["am_0", "am_0_regular"]:
            nsteps = 1
        else:
            nsteps = self.scheme.k

        if "decoupled_sampling" in kwargs:
            f.resample_rff_weights()

        alpha, beta = self.scheme.build_coeffs(timeline)
        N = alpha.shape[0]
        h = timeline[1:] - timeline[0:-1]
        pred = torch.zeros(N, y.shape[1], dtype=torch.float64)
        pred[:nsteps, :] = y

        # HERE CONVERT TO NUMPY THE TENSOR
        pred, h = pred.numpy(), h.numpy()
        alpha, beta = alpha.numpy(), beta.numpy()

        for i in range(nsteps, N):
            y_last = pred[(i - nsteps):i, :]
            alpha_last = alpha[i, :-1].reshape(1, -1)
            beta_last = beta[i, :].reshape(1, -1)
            ynext = self._integration_step(
                f, y_last, h[i-1], alpha_last, beta_last, **kwargs)
            pred[i] = ynext

        pred = torch.from_numpy(pred).clone()
        n = pred.shape[0]

        return pred.cpu().detach(), timeline[:n].cpu().detach()


class ExplicitIntegrator(AbstractIntegrator):

    def __init__(self) -> None:
        super().__init__()
        return

    def _integration_step(self,
                          f: DynamicGP,
                          y: np.ndarray,
                          h: np.ndarray,
                          alpha: np.ndarray,
                          beta: np.ndarray,
                          **kwargs) -> Tensor:
        """Perform an integration step for an explicit numerical scheme
        described by the coefficients alpha and beta.

        Args:
            f (Dynamic_GP): Function describing the dynamics f(t,y) of the
            system to be integrated.
            y (np.ndarray): Past steps of the trajectory.
            h (np.ndarray): Stepsize at time t_{n + k - 1}
            alpha (np.ndarray): Past alpha coefficients.
            beta (np.ndarray): Past beta coefficients.

        Returns:
            np.ndarray: Next step y_{n+k}
        """
        alpha_t = alpha.T
        beta_t = beta.T

        explicit_term = np.sum(np.multiply(y, alpha_t), axis=0)
        f_term = np.zeros((y.shape[1],))
        for i in range(self.steps):
            f_term = f_term + np.multiply(beta_t[i], f(None, y[i, :], kwargs))
        ynext = - explicit_term + h * f_term
        return torch.from_numpy(ynext)

    def integrate_n_steps(self,
                          y: Tensor,
                          timeline: Tensor,
                          f: Optional[DynamicGP] = None,
                          model_state_dict: Optional[Mapping[str, Any]] = None,
                          int_state_dict: Optional[Mapping[str, Any]] = None,
                          **kwargs) -> Tensor:
        torch.set_num_threads(1)

        f = self.init_model(state_dict=model_state_dict, f=f)
        self.load_state_dict(int_state_dict)

        if "decoupled_sampling" in kwargs:
            f.resample_rff_weights()

        alpha, beta = self.scheme.build_coeffs(timeline)
        N = alpha.shape[0]
        h = timeline[1:] - timeline[:-1]
        pred = torch.zeros(N, y.shape[1], dtype=torch.float64)
        pred[:self.steps, :] = y

        # HERE CONVERT TO NUMPY THE TENSOR
        pred, h = pred.numpy(), h.numpy()
        alpha, beta = alpha.numpy(), beta.numpy()

        for i in range(self.steps, N):

            y_last = pred[(i - self.steps):i, :]
            alpha_last = alpha[i, :-1].reshape(1, -1)
            beta_last = beta[i, :].reshape(1, -1)

            ynext = self._integration_step(f,
                                           y_last,
                                           h[i-1],
                                           alpha_last,
                                           beta_last,
                                           **kwargs)
            pred[i] = ynext

        pred = torch.from_numpy(pred).clone()
        n = pred.shape[0]

        return pred.cpu().detach(), timeline[:n].cpu().detach()


class TaylorIntegrator(AbstractIntegrator):
    """Integrator to be used with a TaylorGP and TaylorKernel. Integrate by
    applying Taylor series expansion formula.
    """

    def __init__(self, order: int) -> None:
        super().__init__()
        if not isinstance(order, int):
            raise TypeError
        else:
            order = torch.tensor(order, dtype=torch.int)
        self.register_buffer("order", order, persistent=True)
        self.steps = 1

    def integrate_n_steps(self,
                          y: Tensor,
                          timeline: Tensor,
                          f: Optional[DynamicGP] = None,
                          model_state_dict: Optional[Mapping[str, Any]] = None,
                          int_state_dict: Optional[Mapping[str, Any]] = None,
                          **kwargs) -> Tensor:
        torch.set_num_threads(1)

        f = self.init_model(state_dict=model_state_dict, f=f)
        self.load_state_dict(int_state_dict)

        if "decoupled_sampling" in kwargs:
            f.resample_rff_weights()

        N = timeline.shape[0]
        h = timeline[1:] - timeline[0:-1]
        pred = torch.zeros(N, y.shape[1], dtype=torch.float64)
        pred[0, :] = y

        for i in range(1, N):
            pred[i, :] = self._integration_step(f,
                                                pred[i-1, :].reshape(1, -1),
                                                h[i - 1],
                                                **kwargs)

        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).clone()
        n = pred.shape[0]

        return pred.cpu().detach(), timeline[:n].cpu().detach()

    def _integration_step(self,
                          f: DynamicGP,
                          y: Tensor,
                          h: Tensor,
                          **kwargs) -> Tensor:
        """Apply Taylor series expansion formula a single time.

        Args:
            f (DynamicGP): DynamicGP with the dynamics to integrate
            y (Tensor): Integration node
            h (Tensor): Timestep

        Returns:
            Tensor: _description_
        """
        ds = False
        if "decoupled_sampling" in kwargs:
            ds = kwargs["decoupled_sampling"]

        dynamics = torch.zeros(1, y.shape[1], dtype=torch.float64)
        for i in range(f.num_models):
            if ds:
                pred_list = f.models[i].decoupled_sampling(y, **kwargs)
            else:
                pred_list = f.models[i].posterior_mean(y)

            for k in range(f.models[i].order):
                dynamics[0, i] = dynamics[0, i] + \
                    torch.pow(h, k).div(math.factorial(k + 1)) * pred_list[k]
        return y + h * dynamics


class TrajectorySampler(AbstractIntegrator):
    """Integrator using the scipy library and higher order methods to integrate
    the dynamics learned by the GP.
    """

    def __init__(self) -> None:
        super().__init__()
        self.steps = 1

    def integrate_n_steps(self,
                          y: Tensor,
                          timeline: Tensor,
                          f: Optional[DynamicGP] = None,
                          model_state_dict: Optional[Mapping[str, Any]] = None,
                          int_state_dict: Optional[Mapping[str, Any]] = None,
                          **kwargs) -> Tensor:
        """Integrate the dynamics described by f along timeline
        with initial conditions y. This integrator uses scipy.solve_ivp to
        integrate the dynamics, and the integration can be configured via
        the **kwargs parameter.

        Args:
            f (Dynamic_GP): Function describing the dynamics.
            y0 (Tensor): Initial steps needed for starting
            the integration.
            timeline (Tensor): Timeline over which to integrate.

        Returns:
            Tensor: Approximated dynamics of the system.
        """
        torch.set_num_threads(1)

        f = self.init_model(state_dict=model_state_dict, f=f)
        self.load_state_dict(int_state_dict)

        if "decoupled_sampling" in kwargs:
            f.resample_rff_weights()

        if all(isinstance(model, TaylorGP) for model in f.models):
            # If we are using taylor series, the integrator has to use only the
            # first term f' of the series, namely the dynamics of the system.
            kwargs["index"] = 0

        pred, timeline = ode_solution(f, y, timeline, **kwargs)

        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).clone()

        n = pred.shape[0]

        return pred.cpu().detach(), timeline[:n].cpu().detach()
