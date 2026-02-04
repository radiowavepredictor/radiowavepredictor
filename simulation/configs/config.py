# ===== このファイルでyamlをpython用に変換する =====
from ruamel.yaml import YAML

from common.schema.config import RnnConfig,SaveConfig
from simulation.configs.schema import SimulationConfig

yaml=YAML(typ="safe")
with open("simulation/configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)
    
# ===== この設定変数を使っていく =====
SIMULATION_CFG = SimulationConfig(**cfg['simulation'])
RNN_CFG = RnnConfig(**cfg['model'])
SAVE_CFG = SaveConfig(**cfg["save"])