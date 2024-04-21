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
from pathlib import Path
from typing import List, Optional, Tuple

from configs.global_config import GlobalConfig


def get_configs(config_path: Optional[str]) -> GlobalConfig:
    """Given a path pointing to a config .json file, loads the config object
    and return it.

    Args:
        config_path (Optional[str]): Path to a config file

    Returns:
        GlobalConfig: config object
    """
    if config_path is None:
        global_config = GlobalConfig()
    else:
        global_config = GlobalConfig.parse_file(config_path)
    return global_config


def save_config(config: GlobalConfig,
                config_path: str,
                config_name: Optional[str] = "config.json") -> str:
    """Given a config object and a directory, save the config as a .json file
    on the specified path.

    Args:
        config (GlobalConfig): Configuration object.
        config_path (str): Directory to save the config as a json file.
        config_name (Optional[str]): Name of the config file.

    Returns:
        str: .json file object of the config.
    """
    with Path(os.path.join(config_path, config_name)).open("w") as f:
        f.write(config.json())
    return config.json()


def load_configs(config_path: str,
                 config_filename_list: List[str]
                 ) -> Tuple[List[str], List[GlobalConfig]]:
    """Given a directory, loads all the config files with filename contained
    in config_filename_list.

    Args:
        config_path (str): Directory of the config files.
        config_filename_list (List[str]): List of config files to be loaded.

    Returns:
        Tuple[List[str], List[GlobalConfig]]: filenames (without .json) and
        list of config objects
    """

    if len(config_filename_list) > 0:
        filenames = [s.replace(".json", "") for s in config_filename_list]
    else:
        filenames = ["default"]

    # Get all the config files from a folder, given the config filenames
    config_list = []
    if len(config_filename_list) == 0:
        config_list = [get_configs(None)]

    for i, config_filename in enumerate(config_filename_list):
        config_file_path = os.path.join(config_path, config_filename)
        config_list.append(get_configs(config_file_path))

    return filenames, config_list
