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

from typing import List, Tuple

from torch import Tensor


class DataSplitter():
    """Class providing the methods for cutting and/or splitting train and 
    test data 

    Raises:
        ValueError: The indexes are not consistent

    Returns:
        _type_: _description_
    """
    left_train: int
    right_train: int
    left_test: List[int]
    right_test: List[int]

    def __init__(self,
                 a_train: int,
                 b_train: int,
                 a: List[int],
                 b: List[int]) -> None:

        self.left_train, self.right_train = a_train, b_train

        if len(a) != len(b):
            raise ValueError("The two cut-lists are not of the same length!")

        self.left_test, self.right_test = a, b

    def extract_dataset(self,
                        train_trajectory: Tensor,
                        eval_trajectory: Tensor,
                        train_timeline: Tensor,
                        test_timeline: Tensor
                        ) -> Tuple[List[Tuple], List[Tuple]]:
        """Given noiseless and noisy trajectories, plus timeline,
        split the data according to the boundaries indicated by a and b.

        Args:
            noiseless_trajectory (Tensor): Noiseless trajectory.
            noisy_trajectory (Tensor): Noisy trajectory.
            train_timeline (Tensor): Timeline of the train trajectories
            test_timeline (Tensor): Timeline of the test trajectories

        Returns:
            Tuple[List[Tuple], List[Tuple]]: Tuple with all the splits.
        """
        train_data = [self.extract_trajectory(train_trajectory,
                                              train_timeline,
                                              self.left_train,
                                              self.right_train)]

        eval_data = []
        for i, (left, right) in enumerate(zip(self.left_test, self.right_test)):
            eval_data.append(self.extract_trajectory(
                eval_trajectory, test_timeline, left, right))

        return train_data, eval_data

    def extract_trajectory(self,
                           trajectory: Tensor,
                           timeline: Tensor,
                           left_cut: int = 0,
                           right_cut: int = -1) -> Tuple[Tensor, Tensor]:
        """Given a trajectory and its timeline,
        extract the slice indicated by the interval [left_cut, right_cut]

        Args:
            trajectory (Tensor): Trajectory.
            timeline (Tensor): Timeline of the trajectory
            left_cut (int, optional): Left cut. Defaults to 0.
            right_cut (int, optional): Right cut. Defaults to -1.

        Returns:
            Tuple[Tensor, Tensor]: Tuple with the new trajectory and timeline.
        """
        return (trajectory[left_cut:right_cut, :],
                timeline[left_cut:right_cut])
