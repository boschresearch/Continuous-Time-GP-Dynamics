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


from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, List, Mapping, Tuple

import torch
from models.integrator import AbstractIntegrator
from torch import Tensor


def prediction_batch(model_state_dict: Mapping[str, Any],
                     integrator: AbstractIntegrator,
                     eval_list: List[Tuple[Tensor, Tensor]],
                     method: str,
                     **kwargs
                     ) -> Tuple[List[List[Tuple[Tensor, Tensor]]],
                                List[Tuple[Tensor, Tensor]]]:
    """Using the Dynamic_GP and its integrator, predict by taking the initial
    values of the trajectories in eval_list.

    Args:
        state_dict (Mapping[str, Any]): State dictionary for a dynamicGP
        integrator (AbstractIntegrator): Integrator to be used with the
        DynamicGP
        eval_list (List[Tuple[Tensor, ...]]): List of real evaluation
        trajectories
        method (str): Prediction method

    Returns:
        Tuple[List[Tuple[Tensor, Tensor]], ...]: Lists containing tuples with
        the predicted trajectories and its respective timelines
    """

    pred_list = []
    new_eval_list = []

    if integrator.steps == 0:
        steps = 1
    else:
        steps = integrator.steps

    for i in range(len(eval_list)):
        real_trajectory, real_timeline = eval_list[i]
        y0 = real_trajectory[0:steps].reshape(steps, -1)

        pred = []
        with torch.no_grad():
            num_preds = kwargs["ds_num_preds"] if method == "ds" else 1
            pred = predict_trajectories(model_state_dict,
                                        integrator,
                                        y0,
                                        real_timeline,
                                        method,
                                        num_preds=num_preds,
                                        **kwargs)

        n = pred[0][0].shape[0]
        pred_list.append(pred)
        new_eval_list.append([real_trajectory[:n, :],
                              real_timeline[:n]])

    return pred_list, new_eval_list


def predict_trajectories(model_state_dict: Mapping[str, Any],
                         integrator: AbstractIntegrator,
                         y0: Tensor,
                         timeline: Tensor,
                         method: str,
                         num_preds: int = 1,
                         **kwargs
                         ) -> List[Tuple[Tensor, Tensor]]:
    """Given a gp, predict num_preds trajectories. If decoupled sampling is
     true, regenerate the rff weights at every iteration.

    Args:
        state_dict (Mapping[str, Any]): State_dictionary for a DynamicGP model
        integrator (AbstractIntegrator): Integrator instance
        y0 (Tensor): Initial conditions
        timeline (Tensor): Timeline
        method (str): "ds" for decoupled sampling or "mean" for posterior mean.
        num_preds (Optional[int], optional): Number of predictions. Defaults to
        None.

    Raises:
        ValueError: _description_

    Returns:
        List[Tuple[Tensor, Tensor]]: _description_
    """

    ds = True if method == "ds" else False

    with torch.no_grad():
        integrator_opts = {"decoupled_sampling": ds}
        if "ds_prior_noise" in kwargs:
            integrator_opts["ds_prior_noise"] = kwargs["ds_prior_noise"]

        arg = ()
        args = [arg for i in range(num_preds)]
        res = integrator.integrate_n_steps(y0, timeline, model_state_dict=model_state_dict, int_state_dict=integrator.state_dict(), **integrator_opts)
        func = partial(integrator.integrate_n_steps,
                       y0,
                       timeline,
                       model_state_dict=model_state_dict,
                       int_state_dict=integrator.state_dict(),
                       **integrator_opts)
        traj_list = []
        with ProcessPoolExecutor() as executor:
            result = executor.map(func, args)
            traj_list = list(result)

    return traj_list
