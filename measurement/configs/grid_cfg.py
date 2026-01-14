from ruamel.yaml import YAML
from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW
from itertools import product

yaml=YAML(typ="safe")
with open("measurement/configs/grid_cfg.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)

N_JOBS=cfg['n_jobs'] 
GRID_PARAMS=cfg['params']

'''
GRID_PARAMS = {
    "COURCE":[
        {
            "TRAIN":[1,2,4,5,6,7,8,10,11,12,13,14,15,16],
            "VAL":[3,17],
            "PREDICT":9
        },
    ],
    # --- Model params ---
    "RNN_TYPE": [SimpleRNN],
    "OPTIMIZER": [Adam],
    "INPUT_LEN": [25,50],
    "HIDDEN_NUMS": [
        [20],[24,16],[32,32],[48,48],[64,32]
    ],
    "OUT_STEPS_NUM": [1],
    "BATCH_SIZE": [128],
    "LEARNING_RATE": [0.0003,0.0002,0.0005,0.001],
}
'''
# ===== 直積処理 =====
keys = list(GRID_PARAMS.keys())
values = list(GRID_PARAMS.values())

PARAMS_LIST = [dict(zip(keys, combo)) for combo in product(*values)] 

print(f"全組み合わせ数: {len(PARAMS_LIST)}")
print(PARAMS_LIST[0])
