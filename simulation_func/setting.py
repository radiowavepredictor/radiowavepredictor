import math
import numpy as np
import os
from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW,RMSprop

###生成するフェージングデータに関する設定###
L = 2 # 多重波の数
DELTA_D=0.039 #サンプル間隔[m]
DATA_NUM=1000 #一つのデータセットのデータ数
DATA_SET_NUM = 30 #作成するデータセットの数(測定におけるコース数)
 
K_RICE = 0
R = 1  # 反射波の振幅
#R_0 = np.sqrt(K_RICE*R*R) #直接波の振幅
C = 3e8  # 光速 [m/s]
#F = 3.7e9  # キャリア周波数 [Hz]
F = 1.298e9  # キャリア周波数 [Hz]
LAMBDA_0 = C / F # 波長 [m]（=c/f）

### 学習モデルに関する設定 ### 
USE_RNN_LAYER = SimpleRNN #使用するRNNの種類、layerを作るときに使用するclassを直接指定する
USE_OPTIMIZER = Adam

IN_FEATURES = 1
INPUT_LEN = 50
HIDDEN_NUMS = [8] #隠れ層のユニット数を配列で指定
OUT_STEPS_NUM = 1 
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0003

#作成するモデルのパスと名前(予測でもこれを参照しています)
MODEL_PATH = f"simulation_func/model/{USE_RNN_LAYER.__name__}_{INPUT_LEN}_"+"_".join(map(str, HIDDEN_NUMS))+".keras"

### 予測に関する設定 ###
PREDICT_LEN = 500 #再帰で予測する長さ
PLOT_START = 0
PLOT_RANGE = 100 #グラフとして表示する範囲