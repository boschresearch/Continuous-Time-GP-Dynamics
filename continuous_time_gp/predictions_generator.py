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


import argparse
import os
import shutil

import torch
from configs.data_config import MethodEnum, MocapConfig
from configs.get_configs import GlobalConfig, get_configs
from data.datasets.dynamics_dataset import DynamicsDataset
from factories.data_factory import DataFactory
from factories.dynamicgp_factory import DynamicGPFactory
from factories.integrator_factory import IntegratorFactory
from predictions_generator_mocap import predict_mocap
from train.create_training_data import list_mean, list_var, un_normalize_list
from utils.numerical_schemes import get_scheme
from utils.prediction import prediction_batch
from utils.prediction_utils import get_method_list
from utils.train_utils import get_filename_prefix, load_model, set_seed


def predict(cfg: GlobalConfig,
            seed: int,
            save_dir: str,
            exp_dir: str,
            load_from_config: bool = False) -> None:
    """Make predictions for a specific seed as indicated in the config file.
    Eventually load the trained model from exp_dir store the prediction results
    in save_dir\\seed.

    Args:
        cfg (GlobalConfig): Config object.
        seed (int): Seed to load and predict on.
        save_dir (str): Directory where to store the results. A new path with
        the seed name will be created.
        exp_dir (str): Experiment directory.

    Raises:
        ValueError: If the method does not exist.
        NotImplementedError: _description_
    """
    data_cfg = cfg.data_config
    pred_cfg = cfg.prediction_config
    integrator_cfg_list = pred_cfg.integrator_config
    method = cfg.data_config.method

    integrator_list = [IntegratorFactory.build(
        cfg) for cfg in integrator_cfg_list]

    prediction_methods = get_method_list(pred_cfg)

    if method == "multistep":
        scheme = get_scheme(data_cfg.integration_rule)
    elif method == "taylor":
        scheme = None
    else:
        raise ValueError("Unknown method")

    set_seed(int(seed))
    seed = str(seed)
    seed_path = os.path.join(save_dir, seed)
    exp_seed_path = os.path.join(exp_dir, seed)

    if not os.path.exists(seed_path):
        os.makedirs(seed_path)

    if load_from_config:
        set_seed(0)
        (dataset_train,
         train_list,
         eval_list,
         traj_mean,
         traj_std) = DataFactory.build(data_cfg)
        set_seed(int(seed))

        match data_cfg.method:
            case pred_method if pred_method in MethodEnum:
                gp = DynamicGPFactory.build(cfg.dynamicgp_config,
                                            dataset_train)
            case _:
                raise NotImplementedError

        torch.save(gp.state_dict(),
                   os.path.join(exp_seed_path, "final_model.pt"))
    else:
        # Initialize for the case of experiment_dir != None
        # Load data
        train_list = torch.load(os.path.join(exp_seed_path,
                                             "train_trajectories.pt"))
        eval_list = torch.load(os.path.join(exp_seed_path,
                                            "eval_trajectories.pt"))
        (traj_mean,
         traj_std) = torch.load(os.path.join(exp_seed_path,
                                             "normalization_constants.pt"))

        # Initialize DynamicsDataset
        dataset_train = DynamicsDataset(train_list, method, scheme=scheme)
        gp = load_model(cfg.dynamicgp_config,
                        dataset_train,
                        exp_seed_path,
                        "final_model.pt")
        shutil.copyfile(os.path.join(exp_seed_path, "final_model.pt"),
                        os.path.join(seed_path, "final_model.pt"))

    # NOTE: DATA IS SAVED AS NORMALIZED
    torch.save(train_list, os.path.join(
        seed_path, "train_trajectories.pt"))
    torch.save(eval_list, os.path.join(seed_path, "eval_trajectories.pt"))
    torch.save((traj_mean, traj_std),
               os.path.join(seed_path, "normalization_constants.pt"))

    state_dict = gp.state_dict()

    for integrator, int_cfg in zip(integrator_list, integrator_cfg_list):
        for pred_method in prediction_methods:
            pred_opts = {}

            if pred_method == "ds":
                pred_opts["ds_num_preds"] = pred_cfg.num_ds_trajectories
                pred_opts["ds_prior_noise"] = pred_cfg.ds_prior_noise

            # Predict trajectories
            pred_list, new_eval_list = prediction_batch(state_dict,
                                                        integrator,
                                                        eval_list,
                                                        pred_method,
                                                        **pred_opts)

            # This shouldn't be here for mocap
            unnorm_eval_list = un_normalize_list(new_eval_list,
                                                 traj_mean,
                                                 traj_std)
            # Calculate mean and variance of predicted trajectories
            mean_trajectory = []
            var_trajectory = []
            for k in range(len(pred_list)):
                # Here things get complicated because of the data format
                pred_list[k] = un_normalize_list(pred_list[k],
                                                 traj_mean,
                                                 traj_std)

                # Mean trajectory and variance of the batch
                timeline = pred_list[k][0][1]
                mean_trajectory.append((list_mean(pred_list[k]), timeline))
                var_trajectory.append(list_var(pred_list[k]))

            # Save everything
            prefix = get_filename_prefix(pred_method, int_cfg)

            # Predictions are saved in the unnormalized format
            torch.save({f"{pred_method}": pred_list,
                        "mean_trajectory": mean_trajectory,
                        "var_trajectory": var_trajectory
                        },
                       os.path.join(seed_path, f"{prefix}_prediction.pt"))

            if not isinstance(data_cfg.dataset_config, MocapConfig):
                torch.save(unnorm_eval_list,
                           os.path.join(seed_path,
                                        f"{prefix}_unnorm_eval_trajectories.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir',
                        metavar="Experiment dir",
                        type=str,
                        nargs=1,
                        help="Directory containing the experiment files")
    parser.add_argument('--save_dir',
                        metavar="Save dir",
                        type=str,
                        nargs=1,
                        help="Directory to store the prediction files")
    parser.add_argument('--seed',
                        metavar="Seed to be used",
                        type=int,
                        nargs=1,
                        help="Seed used")

    help_msg = "Load from config and do not search for a saved model"
    parser.add_argument("--load_from_config",
                        action="store_true",
                        help=help_msg)
    args = parser.parse_args()
    args_dict = vars(args)

    load_from_config = False
    if args.load_from_config:
        load_from_config = True

    if os.path.exists(args_dict["experiment_dir"][0]):
        exp_dir = os.path.abspath(args_dict["experiment_dir"][0])
    else:
        raise RuntimeError("Invalid path to experiment directory!")

    if "save_dir" in args_dict:
        if os.path.exists(args_dict["save_dir"][0]):
            save_dir = os.path.abspath(args_dict["save_dir"][0])
        else:
            raise RuntimeError("Save directory does not exist")
    else:
        # If no save_dir is provided, save in the same experiment folder.
        save_dir = exp_dir

    # Parse seed and check it
    seed = args_dict["seed"][0]
    if not isinstance(seed, int):
        raise AssertionError("Seed has to be an integer")

    cfg_path = os.path.join(exp_dir, "config.json")
    cfg = get_configs(cfg_path)

    shutil.copyfile(os.path.join(exp_dir, "config.json"),
                    os.path.join(save_dir, "config.json"))

    if isinstance(cfg.data_config.dataset_config, MocapConfig):
        predict_mocap(cfg, seed, save_dir, exp_dir)
    else:
        predict(cfg, seed, save_dir, exp_dir)
