from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import AdamW
from keras_tuner import HyperModel

from setting import *

class RNNHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(INPUT_LEN, 1)))

        depth=hp.Int("depth",1,3)
        # 層の数・サイズ
        for i in range(depth):
            model.add(USE_RNN_LAYER(units=hp.Int(f"units_{i}", 4, 128, step=4),return_sequences=(i < depth - 1)))

        optimizer = AdamW(learning_rate=LEARNING_RATE)
        model.add(Dense(1, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=optimizer,
        )
        model.summary()
        return model
