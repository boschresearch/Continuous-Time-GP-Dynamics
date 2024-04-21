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
import sys
from datetime import datetime

from utils.train_utils import parse_config_filenames

LOG_FORMAT = '%(asctime)s : %(message)s'
root_dir = os.path.join(sys.path[0], "run_data")

seed_list = [6233, 6331, 6386, 8668, 9415]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--experiment_dir',
                       metavar="Experiment dir",
                       type=str,
                       nargs='+',
                       help="Directories containing the experiment files")
    parser.add_argument('--output_dir',
                        metavar="Output dir",
                        type=str,
                        nargs=1,
                        help="Directory to store the prediction files")
    group.add_argument('--config_path',
                       metavar="Config Path",
                       type=str,
                       nargs='+',
                       help="Paths to the config file")
    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict["config_path"] is None and args_dict["experiment_dir"] is None:
        err = "Neither Experiment directory or Config path exist"
        raise AssertionError(err)

    if args_dict["output_dir"] is None:
        raise RuntimeError("output_dir is None")
    else:
        output_dir = os.path.abspath(args_dict["output_dir"][0])
        if not os.path.isabs(output_dir):
            raise RuntimeError("Output_dir is not an absolute directory")

    sources = []
    folder_names = []
    load_from_cfg = []
    save_dirs = []

    if args_dict["config_path"] is not None:
        # Append every prediction coming from every given config file.
        for path in args_dict["config_path"]:
            if os.path.exists(path):
                cfg_path = os.path.abspath(path)
            else:
                raise RuntimeError("Invalid path to config file")

            # Create directory for predictions
            cfg_name = cfg_path.split(os.sep)[-1]
            filename = parse_config_filenames([cfg_name])[0]
            date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            name = f"{date}_{filename}"

            # Save informations about experiment to be carried on
            save_dir = os.path.join(output_dir, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dirs.append(save_dir)
            sources.append(cfg_path)
            folder_names.append(name)
            load_from_cfg.append(False)

    if args_dict["experiment_dir"] is not None:
        # Append every prediction coming from every given experiment directory.
        for dir in args_dict["experiment_dir"]:
            exp_dir = os.path.abspath(dir)
            if not os.path.exists(exp_dir):
                raise AssertionError(
                    f"Experiment directory {exp_dir} does not exist")
            name = exp_dir.split(os.sep)[-1]

            # Save informations about experiment to be carried on
            save_dir = os.path.join(output_dir, name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dirs.append(save_dir)
            sources.append(exp_dir)
            folder_names.append(name)
            load_from_cfg.append(False)

    iterator = zip(sources, save_dirs, load_from_cfg, folder_names)

    for source, save_dir, load_cfg, name in iterator:

        script_path = os.path.join(sys.path[0], "predictions_generator.py")

        for seed in seed_list:
            command = f"python {script_path} --seed {seed} --save_dir {save_dir}"

            if load_cfg:
                # Initialize the directory in case the config has to be loaded.
                # (and prepare the command line)
                shutil.copyfile(cfg_path, os.path.join(
                    save_dir, "config.json"))
                command = command + " --load_from_config"

            else:  # Directory from where to load the model
                command = command + f" --experiment_dir {source}"

            os.system(command=command)
