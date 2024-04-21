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

"""Collection of functions for building datasets containing the evolution of 
Dynamical systems"""

import sys
from typing import Callable, List, Tuple, Union

import numpy as np
import scipy
import torch
from scipy.integrate import solve_ivp
from torch import Tensor

module = sys.modules[__name__]


def generate_trajectory(model_name: str,
                        y_init: Tensor,
                        timeline: Tensor) -> Tensor:
    """This function performs a numerical simulation for a specific model 
    starting from certain initial conditions.

    Args:
        model_name (str): String indicating which dataset to generate.
        y_init (Tensor): Initial conditions of the simulation.
        timeline (Tensor): Timeline of the simulation. 

    Returns:
        Tensor: Matrix X with training points on each row
    """
    model = getattr(module, model_name)
    X, timeline = ode_solution(model, y_init, timeline)
    return X


def generate_timeline(t_endpoints: Union[List, Tuple],
                      dt: Tensor,
                      bound: Tensor) -> Tensor:
    """Generate a timeline, with even or uneven timesteps that can vary within
     a certain limit described in bound.

    Args:
        t_endpoints (list): Initial and final time of the timeline.
        dt (Tensor): Reference timestep.
        bound (Tensor): Maximum variation on the timestep e.g. 0.05 for a 5% 
        bound.

    Returns:
        Tensor: The generated timeline 
    """
    bound = 2 * (dt*bound)
    max_num_samples = int((t_endpoints[1] - t_endpoints[0])/((1-bound)*dt))
    t_eval = torch.zeros(max_num_samples, dtype=torch.float64)
    t_eval[0] = t_endpoints[0]
    num_samples = 1
    while True:
        time_sample = t_eval[num_samples - 1] + dt + \
            (torch.rand(1).double() - 1/2) * bound
        if time_sample > t_endpoints[1] or num_samples >= max_num_samples:
            break
        t_eval[num_samples] = time_sample
        num_samples += 1
    t_eval = t_eval[:num_samples]

    return t_eval


def add_noise(trajectory: Tensor, noise_std: Tensor) -> Tensor:
    """Adds random gaussian noise to a trajectory.

    Args:
        trajectory (Tensor): Trajectory.
        noise_std (float): Noise standard deviation.

    Returns:
        Tensor: Noisy trajectory
    """
    return trajectory + noise_std * torch.randn(trajectory.shape,
                                                dtype=torch.float64,
                                                device=trajectory.device)


def ode_solution(ode: Callable,
                 y_init: Tensor,
                 t_eval: Tensor,
                 **kwargs) -> Tuple[Tensor, Tensor]:
    """Given an ODE, numerically simulate it in a certain timespan t_span, 
    with a certain dt, and initial conditions y_init.

    Args:
        ode (callable): Function representing the ODE to simulate.
        y_init (Tensor): Tensor of initial conditions.
        t_eval (Tensor): Timeline to simulate [t_0, t_1, ... , t_N]

    Returns:
        Tuple[Tensor, Tensor]: Trajectory and timeline tensors.
    """
    t_span = [t_eval[0], t_eval[-1]]
    y_init = np.squeeze(y_init.numpy())
    sol = solve_ivp(ode, t_span, y_init, t_eval=t_eval, args=(kwargs,))
    return torch.tensor(sol.y).T, torch.tensor(sol.t)


def evaluate_dynamics(x: Tensor, model: str) -> Tensor:
    """Evaluate the dynamics of the requested model at the points X.

    Args:
        X (Tensor): Datapoints to evaluate.
        model (str): Which model to use.

    Returns:
        Tensor: Dynamics evaluation at datapoints X.
    """
    f = getattr(module, model)
    y = torch.zeros(x.shape, dtype=torch.float64)

    for i in range(x.shape[1]):
        y[i, :] = torch.as_tensor(f(0, x[i, :]))

    return y


def exponential_decay(t, y, kwargs=None):
    """ODE for a simple exponential decay."""
    scale = 0.5
    return -scale * y


def van_der_pol(t, z, kwargs=None):
    """ODE for the Van Der Pol oscillator."""
    mu = 0.5
    x, y = z
    return [y, mu*(1 - x**2)*y - x]


def damped_harmonic_oscillator(t, z, kwargs=None):
    """ODE for the Damped Harmonic-Oscillator model."""

    x, y = z
    return [- 0.1*x**3 + 2.0 * y**3, - 2*x**3 - 0.1 * y**3]





