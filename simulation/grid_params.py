from keras.layers import SimpleRNN, LSTM, GRU
from keras.optimizers import Adam, AdamW, RMSprop
from itertools import product

### すべてのグリッド設定を1つにまとめる ###
GRID_PARAMS = {
    # --- Data params ---
    "DELTA_D": [0.005],
    "DATA_NUM": [1000, 2000],
    "DATA_SET_NUM": [10, 50],
    "K_RICE": [0],
    # --- Model params ---
    "RNN_TYPE": [SimpleRNN],
    "OPTIMIZER": [Adam],
    "INPUT_LEN": [50],
    "HIDDEN_NUMS": [[16, 8]],
    "OUT_STEPS_NUM": [1],
    "BATCH_SIZE": [128],
    "LEARNING_RATE": [0.0003],
}

# ===== 直積処理 =====
keys = list(GRID_PARAMS.keys())
values = list(GRID_PARAMS.values())

PARAMS_LIST = [dict(zip(keys, combo)) for combo in product(*values)]

# ===== 結果 =====
print(f"全組み合わせ数: {len(PARAMS_LIST)}")
print(PARAMS_LIST[0])
