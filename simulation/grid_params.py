from keras.layers import SimpleRNN, LSTM, GRU
from keras.optimizers import Adam, AdamW, RMSprop
from itertools import product

N_JOBS=3 # いくつ並列処理させるか

### すべてのグリッド設定を1つにまとめる ###
GRID_PARAMS = {
    # --- Data params ---
    "DELTA_D": [0.005],
    "DATA_NUM": [3000],
    "DATA_SET_NUM": [15],
    "K_RICE": [0,4,8],
    # --- Model params ---
    "RNN_TYPE": [SimpleRNN],
    "OPTIMIZER": [Adam,AdamW],
    "INPUT_LEN": [25,50],
    "HIDDEN_NUMS": [[16, 8],[20],[24,16],[32,32]],
    "OUT_STEPS_NUM": [1],
    "BATCH_SIZE": [32,64,128],
    "LEARNING_RATE": [0.0003,0.0002,0.0005],
}

# ===== 直積処理 =====
keys = list(GRID_PARAMS.keys())
values = list(GRID_PARAMS.values())

PARAMS_LIST = [dict(zip(keys, combo)) for combo in product(*values)]

print(f"全組み合わせ数: {len(PARAMS_LIST)}")
print(PARAMS_LIST[0])
