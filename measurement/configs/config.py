# ===== このファイルでyamlをpython用に変換する =====
from ruamel.yaml import YAML
from pathlib import Path

from common.schema.config import RnnConfig,SaveConfig
from measurement.configs.schema import MeasureConfig

yaml=YAML(typ="safe")
with open(Path("measurement")/"configs"/"config.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)

# ===== この設定定数を使っていく =====
MEASURE_CFG= MeasureConfig(**cfg['measure'])
RNN_CFG = RnnConfig(**cfg['model'])
SAVE_CFG = SaveConfig(**cfg["save"])

