# ===== このファイルでyamlをpython用に変換する =====
from ruamel.yaml import YAML
from pathlib import Path

from common import RnnConfig,SaveConfig
from .schema import SimulationConfig

yaml=YAML(typ="safe")
with open(Path(__file__).parent/"config.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)
    
# ===== この設定変数を使っていく =====
SIMULATION_CFG = SimulationConfig(**cfg['simulation'])
RNN_CFG = RnnConfig(**cfg['model'])
SAVE_CFG = SaveConfig(**cfg["save"])