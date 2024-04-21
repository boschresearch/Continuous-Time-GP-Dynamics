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

import logging
import os
import sys
from datetime import datetime

from parse_arguments import parse_arguments
from utils.train_utils import parse_config_filenames
import subprocess

LOG_FORMAT = '%(asctime)s : %(message)s'
root_path = "run_data"

seed_list = [6233]
 

def main() -> None:
    config_dir, config_filename_list = parse_arguments()
    filenames = parse_config_filenames(config_filename_list)

    folder_names = []
    config_path_list = []

    for i, config_filename in enumerate(config_filename_list):
        config_path_list.append(os.path.join(config_dir, config_filename))

    for i, cfg_path in enumerate(config_path_list):

        date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

        folder_names.append(f"{date}_{filenames[i]}")
        main_path = os.path.join(sys.path[0],
                                 root_path,
                                 folder_names[i])

        if not os.path.exists(main_path):
            os.makedirs(main_path)

        # Set logging
        log_path = os.path.join(main_path, "training_runtime.log")
        logging.basicConfig(filename=log_path,
                            filemode="w",
                            format=LOG_FORMAT,
                            level=logging.INFO)
        logging.info(f"Experiment started for config {filenames[i]}.")

        for _, seed in enumerate(seed_list):
            script_path = os.path.join(sys.path[0], "main_loop.py")
            command = f"python {script_path} " + \
                f"--config_path {cfg_path} --experiment_dir {main_path} " + \
                f" --seed {seed}"
            subprocess.run(command)



if __name__ == "__main__":
    main()
