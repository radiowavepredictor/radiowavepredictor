'''
YAMLファイルでは、文字列しか扱えないので、
文字列→classに変換するためのMAP変数
MAP変数と同じ文字列しか受け取れないようにするために、Enumを用意しています
'''
from enum import Enum
from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW,RMSprop

class RNNType(str,Enum):
    SimpleRNN = "SimpleRNN"
    GRU = "GRU"
    LSTM = "LSTM"
    
RNN_CLASS_MAP = {
    "SimpleRNN": SimpleRNN,
    "GRU": GRU,
    "LSTM": LSTM,
}

class OptimizerType(str,Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    RMSprop = "RMSprop"

OPTIMIZER_MAP = {
    "Adam": Adam,
    "AdamW": AdamW,
    "RMSprop": RMSprop,
}