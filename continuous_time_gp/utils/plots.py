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
from typing import List, Optional, Tuple

import matplotlib.pyplot as pl
import numpy as np
import torch
from matplotlib import colors
from matplotlib.pyplot import Axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch import Tensor

os.environ['MPLCONFIGDIR'] = '/tmp/'

pl.rcParams.update({
    "text.usetex": True
})


def generate_plots(trajectory_mean: List[Tuple[Tensor, Tensor]],
                   trajectory_var: List[Tensor],
                   file_path: str,
                   filename: str,
                   eval_traj: Optional[List[Tuple[Tensor, Tensor]]] = None,
                   train_set: Optional[Tuple[Tensor, Tensor]] = None,
                   **kwargs
                   ) -> None:
    dimensions = trajectory_mean[0][0].shape[1]

    for k in range(len(trajectory_mean)):
        if eval_traj is not None:
            ground_truth = eval_traj[k]
        else:
            ground_truth = None
        trajectory = trajectory_mean[k][0]
        timeline = trajectory_mean[k][1]
        std = torch.sqrt(trajectory_var[k])
        upper = trajectory + 2*std
        lower = trajectory - 2*std

        rollout_plot(timeline,
                     trajectory,
                     upper,
                     lower,
                     file_path,
                     f"{filename}_rollouts_{k}.svg",
                     ground_truth=ground_truth,
                     train_set=train_set)

        if dimensions == 2:
            phase_rollouts_plot(timeline,
                                trajectory,
                                upper,
                                lower,
                                file_path,
                                f"{filename}_phase_rollouts_{k}.svg",
                                ground_truth=ground_truth,
                                train_set=train_set)

            phase_plot_2d(trajectory,
                          file_path,
                          f"{filename}_phase_{k}.svg",
                          ground_truth=ground_truth,
                          train_set=train_set)

        if "mocap" in kwargs and train_set is not None:
            plot_latents_3d(trajectory,
                            timeline,
                            file_path,
                            f"{filename}_latents_{k}.svg",
                            ground_truth=ground_truth[0],
                            train_set=train_set[0])

    return


def generate_local_l2_plot(mean_error: Tensor,
                           var_error: Tensor,
                           timeline: Tensor,
                           file_path: str,
                           filename: str) -> None:
    """Generate the time vs. local l_2 error plot with uncertainty interval 
    of +-2 sigma 

    Args:
        mean_error (Tensor): List with mean l_2 error over time
        var_error_list (Tensor): List with var. of l_2 error over time
        timelines (Tensor): Timeline
        file_path (str): Path 
        filename (str): Filename
    """
    f, ax = pl.subplots(1, 1, figsize=(5, 5))
    ax = [ax]

    std = torch.sqrt(var_error)
    upper = mean_error + 2*std
    lower = mean_error - 2*std

    label = r"$\displaystyle L_2  error$"
    rollout(ax[0],
            timeline,
            mean_error,
            upper=upper,
            lower=lower,
            color="g",
            label=label)
    f.legend()
    save_plot(file_path, filename + ".svg")
    pl.close(f)
    return


def rollout_plot(timeline: Tensor,
                 trajectory: Tensor,
                 upper: Tensor,
                 lower: Tensor,
                 file_path: str,
                 filename: str,
                 ground_truth: Optional[Tuple[Tensor, Tensor]] = None,
                 train_set: Optional[Tuple[Tensor, Tensor]] = None
                 ) -> None:
    """Generate and save the plot for rollouts in every component of the
    provided trajectory.

    Args:
        timeline (Tensor): Timeline
        trajectory (Tensor): Trajectory
        upper (Tensor): Trajectory + 2*std
        lower (Tensor): Trajectory - 2*std
        file_path (str): File path
        filename (str): Filename
        ground_truth (Optional[Tuple[Tensor, Tensor]], optional): Ground truth.
        Defaults to None.
        train_set (Optional[Tuple[Tensor, Tensor]], optional): Train tuple.
        Defaults to None.
    """
    # Since the data is supposed to have at max 3-4 dimensions,
    # there is no need to plot it on different rows
    dimensions = trajectory.shape[1]
    f, ax = pl.subplots(1, dimensions, figsize=(5*dimensions, 5))

    if isinstance(ax, Axes):
        ax = [ax]
    for i in range(dimensions):

        post_str = "Posterior" if i == 0 else ""
        rollout(ax[i],
                timeline,
                trajectory[:, i],
                upper=upper[:, i],
                lower=lower[:, i],
                color="b",
                label=post_str)

        if ground_truth is not None:
            true_trajectory = ground_truth[0]
            true_timeline = ground_truth[1]

            rollout(ax[i],
                    true_timeline,
                    true_trajectory[:, i],
                    color="k",
                    label="Ground Truth" if i == 0 else "")
        # Redundant code
        if train_set is not None:
            train_trajectory = train_set[0]
            train_timeline = train_set[1]

            rollout(ax[i],
                    train_timeline,
                    train_trajectory[:, i],
                    color="k",
                    label="Training Observations" if i == 0 else "",
                    marker="x",
                    linestyle="None")

    f.legend()
    save_plot(file_path, filename)
    pl.close(f)
    return


def phase_rollouts_plot(timeline: Tensor,
                        trajectory: Tensor,
                        upper: Tensor,
                        lower: Tensor,
                        file_path: str,
                        filename: str,
                        ground_truth: Optional[Tuple[Tensor, Tensor]] = None,
                        train_set: Optional[Tuple[Tensor, Tensor]] = None
                        ) -> None:
    """Generate and save the plot for phase portrait along with rollouts in 
    every component of the provided trajectory.

    Args:
        timeline (Tensor): Timeline
        trajectory (Tensor): Trajectory
        upper (Tensor): Trajectory + 2*std
        lower (Tensor): Trajectory - 2*std
        file_path (str): File path
        filename (str): Filename
        ground_truth (Optional[Tuple[Tensor, Tensor]], optional): Ground truth.
        Defaults to None.
        train_set (Optional[Tuple[Tensor, Tensor]], optional): Train tuple.
        Defaults to None.
    """
    dimensions = trajectory.shape[1]
    nfigs = dimensions + 1
    f, ax = pl.subplots(1, nfigs, figsize=(5*nfigs, 5))

    phase(ax[0], trajectory)  # , label="Posterior")

    if ground_truth is not None:
        train_trajectory = ground_truth[0]
        phase(ax[0],
              train_trajectory,
              color="k",
              label="Ground Truth")

    if train_set is not None:
        train_trajectory = train_set[0]
        phase(ax[0],
              train_trajectory,
              color="k",
              label="Training observations",
              marker="x",
              linestyle="None")

    for i in range(1, nfigs):

        post_str = "Posterior" if i == 1 else ""
        rollout(ax[i],
                timeline,
                trajectory[:, i-1],
                upper=upper[:, i-1],
                lower=lower[:, i-1],
                color="b",
                label=post_str)

        if ground_truth is not None:
            train_trajectory = ground_truth[0]
            train_timeline = ground_truth[1]
            rollout(ax[i],
                    train_timeline,
                    train_trajectory[:, i-1],
                    color="k")

        if train_set is not None:
            train_trajectory = train_set[0]
            train_timeline = train_set[1]
            rollout(ax[i],
                    train_timeline,
                    train_trajectory[:, i-1],
                    color="k",
                    marker="x",
                    linestyle="None")

    f.legend()
    save_plot(file_path, filename)
    pl.close(f)
    return


def phase_plot_2d(trajectory: Tensor,
                  file_path: str,
                  filename: str,
                  ground_truth: Optional[Tuple[Tensor, Tensor]] = None,
                  train_set: Optional[Tuple[Tensor, Tensor]] = None
                  ) -> None:
    """Generate and save the plot for phase portrait of the provided trajectory

    Args:
        trajectory (Tensor): Trajectory
        file_path (str): File path
        filename (str): Filename
        ground_truth (Optional[Tuple[Tensor, Tensor]], optional): Ground truth.
        Defaults to None.
        train_set (Optional[Tuple[Tensor, Tensor]], optional): Train tuple.
        Defaults to None.
    """
    f, ax = pl.subplots(1, 1, figsize=(5, 5))

    phase(ax, trajectory, label="Posterior")

    if ground_truth is not None:
        truth_traj = ground_truth[0]
        phase(ax,
              truth_traj,
              color="k",
              label="Ground Truth")

    if train_set is not None:
        train_trajectory = train_set[0]
        phase(ax,
              train_trajectory,
              color="k",
              label="Training observations",
              marker="x",
              linestyle="None")

    f.legend()
    save_plot(file_path, filename)
    pl.close(f)
    return


def phase_plot_3d(trajectory: Tensor,
                  file_path: str,
                  filename: str,
                  ground_truth: Optional[Tuple[Tensor, Tensor]] = None,
                  train_set: Optional[Tuple[Tensor, Tensor]] = None
                  ) -> None:
    """Phase plot for 3D systems.

    Args:
        trajectory (Tensor): 3D trajectory
        file_path (str): File path
        filename (str): Filename
        ground_truth (Optional[Tuple[Tensor, Tensor]], optional): Ground truth.
        Defaults to None.
        train_set (Optional[Tuple[Tensor, Tensor]], optional): Train set.
        Defaults to None.
    """
    raise NotImplementedError
    f, ax = pl.subplots(1, 1, figsize=(5, 5), projection="3d")

    phase(ax, trajectory, label="Posterior")

    if ground_truth is not None:
        truth_traj = ground_truth[0]
        phase(ax,
              truth_traj,
              color="k",
              label="Ground Truth")

    if train_set is not None:
        train_trajectory = train_set[0]
        phase(ax,
              train_trajectory,
              color="k",
              label="Training observations",
              marker="x",
              linestyle="None")

    f.legend()
    save_plot(file_path, filename)
    pl.close(f)
    return


def rollout(ax: Axes,
            timeline: Tensor,
            trajectory: Tensor,
            upper: Optional[Tensor] = None,
            lower: Optional[Tensor] = None,
            color: str = "b",
            label: Optional[str] = "",
            marker: Optional[str] = None,
            linestyle: Optional[str] = "-") -> None:
    """Create a phase plot on the selected figure with the selected
     characteristics.

    Args:
        ax (Axes): Axes indicating the subplot.
        timeline (Tensor): Timeline
        trajectory (Tensor): n-Dimensional-Trajectory to plot
        color (str, optional): Color. Defaults to "b".
        label (Optional[str], optional): Label. Defaults to "".
        marker (Optional[str], optional): Marker. Defaults to None.
        linestyle (Optional[str], optional): Linestyle. Defaults to "-".
    """

    # Not really comprehensible code
    ax.plot(timeline.detach().numpy(),
            trajectory.detach().numpy(),
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=label)

    if upper is not None and lower is not None:
        if label != "" and label is not None:
            post_str = label + r"$\displaystyle \pm 2 \sigma$"
        else:
            post_str = ""

        upper = upper.squeeze()
        lower = lower.squeeze()
        ax.fill_between(timeline.detach().numpy(),
                        upper.numpy(),
                        lower.numpy(),
                        alpha=0.5,
                        label=post_str)

    ax.set(xlabel="Time")
    return


def phase(ax: Axes,
          trajectory: Tensor,
          color: str = "b",
          label: Optional[str] = "",
          marker: Optional[str] = None,
          linestyle: Optional[str] = "-") -> None:
    """Create a phase plot on the selected figure with the selected
     characteristics.

    Args:
        ax (Axes): Axes indicating the subplot.
        trajectory (Tensor): 2D-Trajectory to plot
        color (str, optional): Color. Defaults to "b".
        label (Optional[str], optional): Label. Defaults to "".
        marker (Optional[str], optional): Marker. Defaults to None.
        linestyle (Optional[str], optional): Linestyle. Defaults to "-".
    """
    dimensions = trajectory.shape[1]

    if dimensions == 2:
        ax.plot(trajectory[:, 0].detach().numpy(),
                trajectory[:, 1].detach().numpy(),
                marker=marker,
                label=label,
                color=color,
                linestyle=linestyle)

        ax.set(xlabel=r"$\displaystyle x_1$",
               ylabel=r"$\displaystyle x_2$")

    if dimensions == 3:
        ax.plot(trajectory[:, 0].detach().numpy(),
                trajectory[:, 1].detach().numpy(),
                trajectory[:, 2].detach().numpy(),
                marker=marker,
                color=color,
                linestyle=linestyle)
        ax.set(xlabel=r"$\displaystyle x_1$",
               ylabel=r"$\displaystyle x_2$",
               zlabel=r"$\displaystyle x_3$")
    return


def save_plot(file_path: str, filename: str) -> None:
    """Save a plot as an .svg file on file_path with the indicated filename.
    """
    path_phase = os.path.join(file_path, filename)
    extension = filename.split(".")[-1]
    try:
        pl.savefig(path_phase, dpi=300, format=extension)
        #pl.show()
        #pl.close()
    except RuntimeError:
        return
    return


# MOCAP PLOTS


def plot_latents_3d(trajectory: Tensor,
                    timeline: Tensor =None,
                    file_path: str = None,
                    filename: str = None,
                    ground_truth: Optional[Tensor] = None,
                    train_set: Optional[Tensor] = None) -> None:
    """Function used to plot the first 3 components of the PCA expansion in
     the MoCap dataset.

    Args:
        trajectory (Tensor): Posterior trajectory
        timeline (Tensor): Timeline
        file_path (str): File path
        filename (str): Filename
        ground_truth (Optional[Tensor], optional): Ground truth. 
        Defaults to None
        train_set (Optional[Tensor], optional): Training data. Defaults to None
    """
    trajectory = trajectory.detach().numpy()
    ground_truth = ground_truth.detach().numpy()
    train_set = train_set.detach().numpy()

    f = pl.figure(figsize=(5, 5))
    ax = f.gca(projection='3d')
    cmap = 'gist_rainbow'
#    norm = colors.Normalize(vmin=timeline.min(),
#                            vmax=timeline.max())

    points = np.array([trajectory[:, 0],
                       trajectory[:, 1],
                       trajectory[:, 2]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, cmap=cmap, alpha=0.4, norm=norm)
    lc.set_array(timeline)
    lc.set_linewidth(2)
    ax.add_collection(lc)

    ax.scatter(trajectory[:, 0],
               trajectory[:, 1],
               trajectory[:, 2],
               c='k',
               marker='.',
               s=20,
               zorder=3,
               label="Prediction")

    if ground_truth is not None:
        ax.scatter(ground_truth[:, 0],
                   ground_truth[:, 1],
                   ground_truth[:, 2],
                   c='r',
                   marker='.',
                   s=20,
                   zorder=3,
                   label="Ground truth")

    if train_set is not None:
        ax.scatter(train_set[:, 0],
                   train_set[:, 1],
                   train_set[:, 2],
                   c='k',
                   marker='x',
                   s=20,
                   zorder=3,
                   label="Train trajectory")

    ax.set_xlabel("Comp 1")
    ax.set_ylabel("Comp 2")
    ax.set_zlabel("Comp 3")

    f.legend()
    save_plot(file_path, filename)
    pl.close(f)
    return


def mocap_observation_plots(mean_pred: List[Tuple[Tensor, Tensor]],
                            ground_truth: List[Tuple[Tensor, Tensor]],
                            file_path: str,
                            filename: str) -> None:

    observations = [1, 5, 11, 41, 42, 47]  # [0, 4, 10, 40, 41, 46]
    for num_pred in range(len(mean_pred)):
        f, ax = pl.subplots(2, 3, figsize=(15, 10))
        timeline = mean_pred[0][1].detach().numpy()
        for i, obs_num in enumerate(observations):
            row = i % 2
            col = int(i/2)
            pred_traj = mean_pred[num_pred][0][:, obs_num].detach().numpy()
            eval_traj = ground_truth[num_pred][0][:, obs_num].detach().numpy()
            ax[row][col].plot(timeline,
                              pred_traj,
                              label="Prediction",
                              color="b")
            ax[row][col].plot(timeline,
                              eval_traj,
                              label="Ground truth",
                              color="k")

        save_plot(file_path, filename + f"_{num_pred}.svg")
        pl.close(f)
    return


def loss_plot(loss_history: List[float], file_path: str) -> None:
    """GIven a list of losses and the path, save a loss plot as .svg file.

    Args:
        loss_history (List[float]): List of losses over epochs
        file_path (str): Path where to save the plot.
    """
    n_epochs = len(loss_history)
    epochs = torch.arange(1, n_epochs + 1, 1).detach().numpy()

    f = pl.figure()
    pl.plot(epochs, loss_history, label="Loss history")
    pl.xlabel("Epochs")
    pl.ylabel("Loss")
    save_plot(file_path, "loss.png")
    pl.close(f)
    return
