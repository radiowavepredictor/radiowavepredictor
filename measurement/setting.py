import os
from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW,RMSprop
path = os.path.dirname(__file__)

TRAINING_COURCES=[2,4,5,6,7,8,10,11,12,14,15,16,17] #学習するコース番号群
VALIDATION_COURCES=[3,9] #検証に使うコース番号群
LEARN_MODE="t" #学習データの種類 tなら時間、dなら距離

### 学習モデルに関する設定 ### 
USE_RNN_TYPE = SimpleRNN #使用するRNNの種類、layerを作るときに使用するclassを直接指定する
USE_OPTIMIZER = AdamW

INPUT_LEN = 50
START_CUT_INDEX=0
END_CUT_INDEX=1 # データの中から10n割のデータを使用(先頭から10n割のデータを使用)
HIDDEN_NUMS = [16,8] #隠れ層のユニット数を配列で指定
#IN_FEATURES = ["ReceivedPower[dBm]" ,"1step_diff[dB]"] # csvから読み込む入力特徴量
IN_FEATURES = ["ReceivedPower[dBm]"]
OUT_FEATURES = ["ReceivedPower[dBm]"]
OUT_STEPS_NUM = 1 
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.001

#作成するモデルのパスと名前(予測でもこれを参照しています)
MODEL_PATH = path+f"/model/simulation_fading_{USE_RNN_TYPE.__name__}_{INPUT_LEN}_"+"_".join(map(str, HIDDEN_NUMS))+".keras"

### 予測に関する設定 ###
PREDICT_COURCE=15 #予測したいコース番号
PREDICT_LEN = 500 #再帰で予測する長さ
PLOT_START = 199
PLOT_RANGE = 40 #グラフとして表示する範囲
