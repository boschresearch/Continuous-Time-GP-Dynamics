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


from configs.prediction_config import PredictionConfig
from typing import List


def get_method_list(prediction_cfg: PredictionConfig) -> List[str]:
    """Read the config for predictions and extract a list with the prediction
    methods.

    Args:
        prediction_cfg (PredictionConfig): Config for predictions.

    Returns:
        List[str]: List with prediction methods.
    """
    prediction_methods = []

    if prediction_cfg.ds_pred:
        prediction_methods.append("ds")
    if prediction_cfg.mean_pred:
        prediction_methods.append("mean")
    return prediction_methods
