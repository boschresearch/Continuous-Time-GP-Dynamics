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
import sys
from typing import List, Tuple


def parse_arguments() -> Tuple[str, List[str]]:
    """Parse the arguments for the main function.

    Returns:
        Tuple[str, List[str]]: Config path and list of config filenames.
    """
    parser = argparse.ArgumentParser()

    default_path = [os.path.join(sys.path[0],
                                 os.path.join("data", "experiment_configs"))]

    parser.add_argument('--config_path',
                        metavar="Config Path",
                        type=str,
                        nargs=1,
                        default=default_path,
                        help="Absolute path containing the config files")
    parser.add_argument('--config_files',
                        metavar="Config Files",
                        type=str,
                        nargs="*",
                        help="Name of the config files (contained in the \
                            config path) to use")

    # parser.add_argument('--seeds',
    #                     metavar="Seeds for trajectories",
    #                     type=int,
    #                     nargs="*",
    #                     help="Seeds to use for every training loop")

    args = parser.parse_args()
    args_dict = vars(args)

    config_path = os.path.abspath(args_dict["config_path"][0])

    if not os.path.exists(config_path):
        raise AssertionError(f"Config path {config_path} does not exist")

    config_filename_list = args_dict["config_files"]
    # seed_list = args_dict["seeds"]

    # if seed_list is None:
    #     seed_list = []
    if config_filename_list is None:
        config_filename_list = []

    return config_path, config_filename_list  # , seed_list
