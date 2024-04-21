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


import sys
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

module = sys.modules[__name__]


def create_omega(timeline: Tensor) -> Tensor:
    """Given the timeline, generate the corresponding omega_t = h_t / h_{t-1}

    Args:
        timeline (Tensor): Timeline of the data.

    Returns:
        Tensor: Vector with Omega values.
    """
    if type(timeline) is not Tensor:
        raise TypeError("Timeline must be a Tensor!")
    if timeline is None:
        raise RuntimeError("Timeline cannot be None!")

    num_patterns = timeline.shape[0]
    h = timeline[1:] - timeline[:-1]
    omega = torch.zeros(num_patterns - 1, dtype=torch.float64)
    for i in range(1, (num_patterns - 1)):
        omega[i] = h[i].div(h[i-1])
    return omega[1:]


def divided_diff(timeline: Tensor, k: int) -> Tensor:
    """Calculates the divided differences coefficients needed for the
    generation of variable stepsize multistep coefficients.
    """

    nsteps = timeline.shape[0]
    coeff = torch.zeros((k + 1, nsteps, nsteps), dtype=torch.float64)
    # the first column is y
    coeff[0, :, :] = torch.eye(nsteps)

    for j in range(1, k + 1):
        for n in range(j, nsteps):
            coeff[j, n, :] = (coeff[j-1, n, :] - coeff[j-1, n-1, :]) / \
                (timeline[n]-timeline[n-j])

    return coeff


def beta_coeff(timeline: Tensor) -> Tensor:
    """Calculates beta coefficients needed for the generation of variable
    stepsize multistep coefficients

    Args:
        timeline (Tensor): Timeline

    Returns:
        Tensor: _description_
    """
    num_steps = timeline.shape[0]
    beta = torch.zeros(num_steps, num_steps)
    beta[0, :] = 1
    for j in range(1, num_steps):
        for n in range(j, num_steps - 1):
            numerator = timeline[n + 1] - timeline[n - j + 1]
            denominator = timeline[n] - timeline[n - j]
            beta[j, n] = beta[j - 1, n] * numerator.div(denominator)
    return beta


def g_coeff(timeline: Tensor, k: int) -> Tensor:
    """Calculates c coefficients needed for the generation of variable
    stepsize multistep coefficients

    Args:
        timeline (Tensor): Timeline

    Returns:
        Tensor: _description_
    """
    h = timeline[1:] - timeline[:-1]
    nsteps = timeline.shape[0]
    c = torch.zeros(k, k, nsteps)

    if k == 1:
        # In case if Euler methods, the coeffs are just 1,
        # so we can skip the computation
        return torch.ones((1, nsteps))

    for n in range(nsteps):
        c[0, 1:, n] = torch.ones(1, k - 1).div(torch.arange(1, k, 1))

        c[1, 1:, n] = torch.ones(1, k - 1).div(
            torch.arange(1, k, 1)*torch.arange(2, k + 1, 1))

    # Maybe n should be translated by 1
    for j in range(2, k):
        for q in range(1, k - j + 1):
            for n in range(j - 1, nsteps):
                c[j, q, n] = c[j - 1, q, n] - c[j - 1, q + 1, n] * \
                    (h[n-1].div(timeline[n] - timeline[n-j]))
    return c[:, 1, :]


def get_phi(k: int, nsteps: int, b: Tensor) -> tuple[Tensor, Tensor]:

    phi = torch.zeros((k, nsteps, nsteps), dtype=torch.float64)
    phi_star = torch.zeros((k, nsteps, nsteps),
                           dtype=torch.float64)  # And this

    phi[0, :, :] = torch.eye(nsteps)
    phi_star[0, :, :] = torch.eye(nsteps)

    for j in range(1, k):
        p_star = torch.cat(
            (torch.zeros(1, nsteps), phi_star[j - 1, :-1, :]), axis=0)
        phi[j, :, :] = phi[j - 1, :, :] - p_star
        phi_star[j, :, :] = b[j, :]*phi[j, :, :]

    return phi, phi_star


# NUMERICAL SCHEMES

class NumericalScheme(ABC):

    k: int = NotImplemented
    name: str = NotImplemented

    @classmethod
    @abstractmethod
    def build_coeffs(cls, timeline: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @classmethod
    def get_k(cls) -> int:
        return cls.k
    
    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def set_k(cls, value: int) -> None:
        if isinstance(value, int) is not True:
            raise TypeError("Number k of steps is not an integer.")
        cls.k = value
        name_scheme = "".join(cls.__name__.split("_")[:-1])
        cls.name = f"{name_scheme}_{str(cls.k)}"
        return


# General Methods for k-order coefficients


class ab_k(NumericalScheme):

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> Tuple[Tensor, Tensor]:

        nsteps = timeline.shape[0]

        g = g_coeff(timeline, cls.k)
        b = beta_coeff(timeline)
        b = torch.cat(
            (torch.ones(cls.k - 1, nsteps), b), axis=0)
        _, phi_star = get_phi(cls.k, nsteps, b)
        f_coeffs = torch.zeros(nsteps, nsteps)

        for j in range(cls.k):
            for n in range(nsteps):
                f_coeffs[n, :] = f_coeffs[n, :] + g[j, n] * phi_star[j, n, :]

        # Create alpha coefficients
        alpha = torch.zeros(nsteps - cls.k, cls.k + 1, dtype=torch.float64)
        alpha[:, cls.k] = torch.ones(nsteps - cls.k, dtype=torch.float64)
        alpha[:, cls.k - 1] = -torch.ones(nsteps - cls.k, dtype=torch.float64)

        # Create beta coefficients
        beta = torch.zeros((nsteps - cls.k, cls.k + 1), dtype=torch.float64)
        for i in range(nsteps - cls.k):
            beta[i, :] = f_coeffs[i + cls.k - 1, i:(i + cls.k + 1)]

        return alpha, beta


class am_k(NumericalScheme):

    k: int

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> Tuple[Tensor, Tensor]:
        """Given the timeline, generates the coefficients for an Adams-Moulton
        method of order k for an arbitrary timeline. Works also for irregular
        timelines.

        Args:
            timeline (Tensor): Timeline

        Returns:
            Tuple[Tensor, Tensor, int]: _description_
        """
        nsteps = timeline.shape[0]
        g = g_coeff(timeline, cls.k + 1)
        b = beta_coeff(timeline)
        b = torch.cat(
            (torch.ones(cls.k, nsteps), b), axis=0)
        phi, phi_star = get_phi(cls.k + 1, nsteps, b)
        f_coeffs = torch.zeros(nsteps, nsteps)

        for j in range(cls.k):
            for n in range(nsteps):
                # This should be generalized to irregular timelines
                # g[j,j] does not hold for them
                f_coeffs[n, :] = f_coeffs[n, :] + g[j, n] * phi_star[j, n, :]

        new_f_coeffs = f_coeffs.clone()
        for n in range(cls.k, nsteps):
            new_f_coeffs[n, :] = f_coeffs[n - 1, :] + \
                g[cls.k, n] * phi[cls.k, n, :]
        f_coeffs = new_f_coeffs.clone()

        # To make sense dimension
        if cls.k == 0:
            k = 1
        else:
            k = cls.k

        # Create alpha coefficients
        alpha = torch.zeros(nsteps - k, k + 1, dtype=torch.float64)
        alpha[:, k] = torch.ones(nsteps - k, dtype=torch.float64)
        alpha[:, k - 1] = -torch.ones(nsteps - k, dtype=torch.float64)

        # Create beta coefficients
        beta = torch.zeros(nsteps - k, k + 1, dtype=torch.float64)
        for i in range(nsteps - k):
            beta[i, :] = f_coeffs[i + k, i:(i + k + 1)]

        return alpha, beta


class bdf_k(NumericalScheme):

    k: int

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> Tuple[Tensor, Tensor]:
        """Given the timeline, generates the coefficients for a Backward
        Difference Formula method of order k for an arbitrary timeline.
        Works also for irregular timelines.

        Args:
            timeline (Tensor): Timeline

        Returns:
            Tuple[Tensor, Tensor]: Alpha and Beta coefficients of the
            correspondent multistep method.
        """
        nsteps = timeline.shape[0]
        h = timeline[1:] - timeline[:-1]

        div_diff = divided_diff(timeline, cls.k)
        delta = torch.ones(cls.k + 1, nsteps)
        coeff = torch.zeros(nsteps, nsteps)

        for i in range(1, cls.k + 1):
            for n in range(i, nsteps):
                delta[i, n - 1] = delta[i - 1, n - 1] * \
                    (timeline[n] - timeline[n - i])
        for j in range(1, cls.k + 1):
            for n in range(cls.k - 1, nsteps):
                coeff[n, :] = coeff[n, :] + \
                    delta[j - 1, n - 1] * div_diff[j, n, :]

        coeff[1:, :] = coeff[1:, :].mul(h.reshape(-1, 1))

        # Create alpha coefficients
        alpha = torch.zeros(nsteps - cls.k, cls.k + 1, dtype=torch.float64)
        for n in range(nsteps - cls.k):
            alpha[n, :] = coeff[n + cls.k, n:(n + cls.k + 1)]

        # Create beta coefficients
        beta = torch.zeros(nsteps - cls.k, cls.k + 1, dtype=torch.float64)
        beta[:, cls.k] = torch.ones(nsteps - cls.k, dtype=torch.float64)

        # n+1-th Coefficient Normalization
        normalization = alpha[:, cls.k].clone()
        for order in range(cls.k + 1):
            alpha[:, order] = alpha[:, order].div(normalization)
            beta[:, order] = beta[:, order].div(normalization)

        return alpha, beta

# Aliases for Euler methods


class forward_euler(NumericalScheme):

    k: int = 1
    name: str = "forward_euler"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ y_{n+1} = y_n + h_n * f(t_n,y_n)

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor, int]: Integration coefficients for Forward 
            Euler method.
        """
        num_steps = timeline.shape[0] - 1

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 0] = - torch.ones(num_steps)
        alpha[:, 1] = torch.ones(num_steps)

        beta = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 0] = torch.ones(num_steps, dtype=torch.float64)

        return alpha, beta


class backward_euler(NumericalScheme):

    k: int = 1
    name: str = "backward_euler"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ y_{n+1} = y_n + h_n * f(t_{n+1},y_{n+1})

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for Backward
            Euler method.
        """
        num_steps = timeline.shape[0] - 1

        alpha = torch.zeros(num_steps, cls.k+1, dtype=torch.float64)
        alpha[:, 0] = - torch.ones(num_steps)
        alpha[:, 1] = torch.ones(num_steps)

        beta = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 1] = torch.ones(num_steps, dtype=torch.float64)

        return alpha, beta

# ADAMS METHODS
# ADAMS MOULTON METHODS FOR REGULAR TIMELINES


class am_0_regular(NumericalScheme):

    k: int = 0
    name: str = "am_0_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ AM1 for timelines with constant stepsize.

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for AB3 method.
        """

        num_steps = timeline.shape[0] - 1

        alpha = torch.zeros(num_steps, 2, dtype=torch.float64)
        alpha[:, 1] = torch.ones(num_steps)
        alpha[:, 0] = - torch.ones(num_steps)

        beta = torch.zeros(num_steps, 2, dtype=torch.float64)
        beta[:, 1] = torch.ones(num_steps)

        return alpha, beta


class am_1_regular(NumericalScheme):

    k: int = 1
    name: str = "am_1_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ AM1 for timelines with constant stepsize.

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for AB3 method.
        """

        num_steps = timeline.shape[0] - cls.k

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 1] = torch.ones(num_steps)
        alpha[:, 0] = - torch.ones(num_steps)

        beta = torch.ones(num_steps, cls.k + 1, dtype=torch.float64).div(2)

        return alpha, beta


class am_2_regular(NumericalScheme):

    k: int = 2
    name: str = "am_2_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ AM2 for timelines with constant stepsize.

            Args:
                timeline (Tensor): Timeline.

            Returns:
                tuple[Tensor, Tensor]: Integration coefficients for AB3 method.
            """
        num_steps = timeline.shape[0] - cls.k

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 2] = torch.ones(num_steps)
        alpha[:, 1] = - torch.ones(num_steps)

        beta = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 2] = torch.ones(num_steps).mul(5/12)
        beta[:, 1] = torch.ones(num_steps).mul(8/12)
        beta[:, 0] = torch.ones(num_steps).mul(- 1/12)

        return alpha, beta

# ADAMS BASHFORTH METHODS FOR REGULAR TIMELINES


class ab_1_regular(NumericalScheme):

    k: int = 1
    name: str = "ab_1_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ AB1 for timelines with constant stepsize.

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for AB3 method.
        """
        num_steps = timeline.shape[0] - cls.k

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 1] = torch.ones(num_steps)
        alpha[:, 0] = - torch.ones(num_steps)

        beta = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 0] = torch.ones(num_steps)

        return alpha, beta


class ab_2_regular(NumericalScheme):

    k: int = 2
    name: str = "ab_2_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ AB2 for timelines with constant stepsize.

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for AB2 method.
        """

        num_steps = timeline.shape[0] - cls.k

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 2] = torch.ones(num_steps)
        alpha[:, 1] = -torch.ones(num_steps)
        alpha[:, 0] = torch.zeros(num_steps)

        beta = torch.ones(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 1] = torch.ones(num_steps).mul(3/2)
        beta[:, 0] = torch.ones(num_steps).mul(- 1/2)

        return alpha, beta


class ab_3_regular(NumericalScheme):

    k: int = 3
    name: str = "ab_3_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ AB3 for timelines with constant stepsize.

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for AB3 method.
        """

        num_steps = timeline.shape[0] - cls.k

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 3] = torch.ones(num_steps)
        alpha[:, 2] = -torch.ones(num_steps)

        beta = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 2] = torch.ones(num_steps).mul(23/12)
        beta[:, 1] = torch.ones(num_steps).mul(- 16/12)
        beta[:, 0] = torch.ones(num_steps).mul(5/12)

        return alpha, beta

# BACKWARD DIFFERENTIATION FORMULAS


class bdf_1_regular(NumericalScheme):

    k: int = 1
    name: str = "bdf_1_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ BDF1 for timelines with constant stepsize.

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for BDF1 method.
        """

        num_steps = timeline.shape[0] - cls.k

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 1] = torch.ones(num_steps)
        alpha[:, 0] = -torch.ones(num_steps)

        beta = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 1] = torch.ones(num_steps)

        return alpha, beta


class bdf_2_regular(NumericalScheme):

    k: int = 2
    name: str = "bdf_2_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ BDF2 for timelines with constant stepsize.

        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for BDF2 method.
        """

        num_steps = timeline.shape[0] - cls.k

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 2] = torch.ones(num_steps)
        alpha[:, 1] = torch.ones(num_steps).mul(- 4/3)
        alpha[:, 0] = torch.ones(num_steps).mul(1/3)

        beta = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 2] = torch.ones(num_steps).mul(2/3)

        return alpha, beta


class bdf_3_regular(NumericalScheme):

    k: int = 3
    name: str = "bdf_3_regular"

    @classmethod
    def build_coeffs(cls, timeline: Tensor) -> tuple[Tensor, Tensor]:
        """ BDF3 for timelines with constant stepsize.
        Args:
            timeline (Tensor): Timeline.

        Returns:
            tuple[Tensor, Tensor]: Integration coefficients for BDF3
            method.
        """

        num_steps = timeline.shape[0] - cls.k

        alpha = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        alpha[:, 3] = torch.ones(num_steps)
        alpha[:, 2] = torch.ones(num_steps).mul(- 18/11)
        alpha[:, 1] = torch.ones(num_steps).mul(9/11)
        alpha[:, 0] = torch.ones(num_steps).mul(-2/11)

        beta = torch.zeros(num_steps, cls.k + 1, dtype=torch.float64)
        beta[:, 3] = torch.ones(num_steps).mul(6/11)

        return alpha, beta


# General Scheme retrieval method

def get_scheme(integration_rule: str
               ) -> NumericalScheme:
    """Given the integration rule, returns the function used to generate
    the coefficients.

    Args:
        integration_rule (str): String indicating the integration rule.

    Returns:
        callable: Coefficient generator.
    """
    if integration_rule is None:
        raise ValueError("integration_rule cannot be None.")

    integration_rule = integration_rule.lower()
    tokens = integration_rule.split("_")
    if len(tokens) == 2:
        if (tokens[0] in ["ab", "am", "bdf"]) and tokens[1].isdigit():
            scheme = tokens[0]
            order = int(tokens[1])
            scheme = getattr(module, "_".join([scheme, "k"]))
            scheme.set_k(order)
            return scheme
    else:
        try:
            return getattr(module, integration_rule)
        except AttributeError:
            raise NotImplementedError(
                f"Method {integration_rule} has not been implemented yet!")
