# ===== このファイルでyamlをpython用に変換する =====
from ruamel.yaml import YAML

from common.registory import RNN_CLASS_MAP,OPTIMIZER_MAP
from measurement.configs.measure_schema import RnnConfig,SaveConfig,MeasureConfig

yaml=YAML(typ="safe")
with open("measurement/configs/measure.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)

rnn_class = RNN_CLASS_MAP[cfg["model"]["rnn_type"]]
optimizer_class = OPTIMIZER_MAP[cfg["model"]["optimizer"]]

# ===== この設定定数を使っていく =====
MEASURE_CFG= MeasureConfig(**cfg['measure'])

RNN_CFG = RnnConfig(
    rnn_class=rnn_class,
    optimizer_class=optimizer_class,
    **{k: v for k, v in cfg["model"].items()   # rnn_とoptimizerの設定だけはyamlから読み取らない
        if k not in ("rnn_type", "optimizer")}
)

SAVE_CFG = SaveConfig(**cfg["save"])