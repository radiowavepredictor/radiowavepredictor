# ===== このファイルでyamlをpython用に変換する =====
from ruamel.yaml import YAML
from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW,RMSprop

from simulation.configs.cfg_schema import FadingConfig,RnnConfig,SaveConfig

RNN_CLASS_MAP = {
    "SimpleRNN": SimpleRNN,
    "GRU": GRU,
    "LSTM": LSTM,
}

OPTIMIZER_MAP = {
    "Adam": Adam,
    "AdamW": AdamW,
    "RMSprop": RMSprop,
}

yaml=YAML(typ="safe")
with open("simulation/configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)
    
rnn_class = RNN_CLASS_MAP[cfg["model"]["rnn_type"]]
optimizer_class = OPTIMIZER_MAP[cfg["model"]["optimizer"]]

# ===== この設定変数を使っていく =====
FADING_CFG = FadingConfig(**cfg['fading'])

RNN_CFG = RnnConfig(
    rnn_class=rnn_class,
    optimizer_class=optimizer_class,
    **{k: v for k, v in cfg["model"].items()   # rnn_とoptimizerの設定だけはyamlから読み取らない
        if k not in ("rnn_type", "optimizer")}
)

SAVE_CFG = SaveConfig(**cfg["save"])