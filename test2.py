import os
import numpy as np
from ruamel.yaml import YAML
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from common.function.function import mw_to_dbm, predict,to_yaml_safe,array_of_array_to_dataset
from common.function.save import save_predict_data
from common.schema import RnnConfig, SaveConfig
from simulation.configs.schema import SimulationConfig
from simulation.function import calc_nakagami_rice_fading
from simulation.configs.config import SIMULATION_CFG

fading_data = calc_nakagami_rice_fading(SIMULATION_CFG)
power = np.abs(fading_data) ** 2
power_db = 10 * np.log10(power)
print(power_db)

predict_result_fig = plt.figure()


plt.plot(power_db)

plt.legend()
plt.show()

