from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW,RMSprop

from common.common_setting import *
from simulation.schema_setting import FadingConfig,RnnConfig,SaveConfig

### 生成するフェージングデータに関する設定 ###
DATA_NUM=3000 #一つのデータセットのデータ数
DATA_SET_NUM = 15 #作成するデータセットの数(測定におけるコース数)
L = 30 # 多重波の数
DELTA_D=0.005 #サンプル間隔[m]

C = 3e8  # 光速 [m/s]
F = 1.298e9  # キャリア周波数 [Hz]
LAMBDA_0 = C / F # 波長 [m]（=c/f）
R = 1  # 反射波の振幅
K_RICE = 40
#R_0 = np.sqrt(K_RICE*R*R) #直接波の振幅

### 学習モデルに関する設定 ### 
RNN_TYPE = SimpleRNN #使用するRNNの種類、layerを作るときに使用するclassを直接指定する
OPTIMIZER = Adam

IN_FEATURES = 1
INPUT_LEN = 50
HIDDEN_NUMS = [16,8] #隠れ層のユニット数を配列で指定
OUT_STEPS_NUM = 1 #何点予測するか
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.0003

### 予測、保存に関する設定 ###
PREDICTED_DATASET_NUM = 3 # RMSEを出すときに複数のデータセットを予測して平均値を求めるが、そのときいくつのデータセットを使用するか
RECURSIVE_NUM = 500 #再帰予測で予測する回数(現状使ってない)

### ↑↑を構造体にまとめる ###
FADING_CFG = FadingConfig(
    l=L,
    data_num=DATA_NUM,
    data_set_num=DATA_SET_NUM,
    delta_d=DELTA_D,
    lambda_0=LAMBDA_0,
    r=R,
    k_rice=K_RICE
)

RNN_CFG = RnnConfig(
    rnn_class=RNN_TYPE,
    optimizer_class=OPTIMIZER,
    in_features=IN_FEATURES,
    out_steps_num=OUT_STEPS_NUM,
    input_len=INPUT_LEN,
    hidden_nums=HIDDEN_NUMS,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
)

SAVE_CFG = SaveConfig(
    plot_start=PLOT_START,
    plot_range=PLOT_RANGE,

    experiment_name=EXPERIMENT_NAME,
    run_name=RUN_NAME,
    use_mlflow=USE_MLFLOW,
    save_dir=SAVE_DIR,
    
    predicted_dataset_num=PREDICTED_DATASET_NUM,
    recursive_num=RECURSIVE_NUM
)