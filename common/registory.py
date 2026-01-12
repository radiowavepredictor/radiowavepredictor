from keras.layers import SimpleRNN,LSTM,GRU
from keras.optimizers import Adam,AdamW,RMSprop

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