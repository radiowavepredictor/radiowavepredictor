from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW
from itertools import product

N_JOBS=2 # いくつ並列処理させるか

GRID_PARAMS = {
    # --- Data params ---
    "DELTA_D": [0.005],
    "DATA_NUM": [4000],
    "DATA_SET_NUM": [17],
    "K_RICE": [4],
    # --- Model params ---
    "RNN_TYPE": [SimpleRNN],
    "OPTIMIZER": [Adam],
    "INPUT_LEN": [10,50],
    "HIDDEN_NUMS": [[20],[16,8]],
    "OUT_STEPS_NUM": [1],
    "BATCH_SIZE": [256],
    "LEARNING_RATE": [0.0005],
}
# ===== 直積処理 =====
keys = list(GRID_PARAMS.keys())
values = list(GRID_PARAMS.values())

PARAMS_LIST = [dict(zip(keys, combo)) for combo in product(*values)] 

print(f"全組み合わせ数: {len(PARAMS_LIST)}")
print(PARAMS_LIST[0])
