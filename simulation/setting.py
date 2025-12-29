from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW,RMSprop
from datetime import datetime,timezone,timedelta
import os

### 生成するフェージングデータに関する設定###
L = 30 # 多重波の数
DELTA_D=0.005 #サンプル間隔[m]
DATA_NUM=1012 #一つのデータセットのデータ数
DATA_SET_NUM = 10 #作成するデータセットの数(測定におけるコース数)
 
K_RICE = 0
R = 1  # 反射波の振幅
#R_0 = np.sqrt(K_RICE*R*R) #直接波の振幅
C = 3e8  # 光速 [m/s]
F = 1.298e9  # キャリア周波数 [Hz]
LAMBDA_0 = C / F # 波長 [m]（=c/f）

### 学習モデルに関する設定 ### 
RNN_TYPE = SimpleRNN #使用するRNNの種類、layerを作るときに使用するclassを直接指定する
OPTIMIZER_TYPE = Adam

IN_FEATURES = 1
INPUT_LEN = 50
HIDDEN_NUMS = [16,8] #隠れ層のユニット数を配列で指定
OUT_STEPS_NUM = 1 
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.0003

#作成するモデルのパスと名前(予測でもこれを参照しています)
#MODEL_PATH = f"simulation_func/model/{USE_RNN_LAYER.__name__}_{INPUT_LEN}_"+"_".join(map(str, HIDDEN_NUMS))+".keras"

### 予測に関する設定 ###
PLOT_START = 0
PLOT_RANGE = 200 #グラフとして表示する範囲
PREDICT_LEN = 500 #再帰で予測する長さ

### mlflow(実験データ管理ツール)における設定 ###
EXPERIMENT_NAME="simulation"
JST = timezone(timedelta(hours=9))
RUN_NAME = datetime.now(JST).strftime("%Y_%m_%d_%H_%M")
USE_MLFLOW=True #mlflowを使うかどうか(mlflowが使えない環境ではFalseを指定)

### mlflowを使わないでjsonで保存するとき ###
BASE_DIR = "exp_runs"
SAVE_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, RUN_NAME)