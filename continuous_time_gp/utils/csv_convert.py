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

import pandas as pd
import argparse
import os
import numpy as np
from typing import List


def csv_to_npz(csv_paths: List[str], output_path: str) -> None:
    """Given a list of paths to csv files containing training data, convert
    them in the format used by this framework.
    """
    for path in csv_paths:
        filename = path.split(os.sep)[-1]
        filename = filename.split(".")[-2]
        df = pd.read_csv(path, delimiter=';', dtype=np.float64, decimal=",")
        df = df.to_numpy()

        timeline = df[:, 0].astype(np.float64)
        noisy_traj = df[:, 1:].astype(np.float64)

        file_path = os.path.join(output_path, filename + ".npz")
        np.savez(file_path,
                 noisy_trajectory=noisy_traj,
                 timeline=timeline,
                 allow_pickle=True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--csv_path',
                       metavar="Experiment dir",
                       type=str,
                       nargs='+',
                       help="Path to the csv files")
    parser.add_argument('--output_dir',
                        metavar="Output dir",
                        type=str,
                        nargs=1,
                        help="Directory to store the .npz files")

    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict["csv_path"] is None:
        err = "CSV path cannot be empty"
        raise AssertionError(err)
    else:
        csv_paths = args_dict["csv_path"]

    if args_dict["output_dir"] is None:
        raise RuntimeError("output_dir is None")
    else:
        output_dir = os.path.abspath(args_dict["output_dir"][0])
        if not os.path.isabs(output_dir):
            raise RuntimeError("Output_dir is not an absolute directory")

    csv_to_npz(csv_paths, output_dir)
