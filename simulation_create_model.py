import matplotlib.pyplot as plt
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

from simulation_func.func import *

dataset,val_dataset=load_fading_data(BATCH_SIZE,INPUT_LEN)
print(val_dataset)

# モデル構築
print("🚀 新しいモデルを作成します")
model = Sequential()
model.add(Input(shape=(INPUT_LEN, IN_FEATURES)))

for hidden_num in HIDDEN_NUMS[:-1]:
    model.add(USE_RNN_LAYER(hidden_num, return_sequences=True))
    #model.add(USE_RNN_LAYER(hidden_num, return_sequences=True,kernel_regularizer=l2(1e-5)))
model.add(USE_RNN_LAYER(HIDDEN_NUMS[-1], return_sequences=False))
#model.add(USE_RNN_LAYER(HIDDEN_NUMS[-1], return_sequences=False,kernel_regularizer=l2(1e-5)))
model.add(Dense(OUT_STEPS_NUM))
model.add(Activation("linear"))
optimizer = USE_OPTIMIZER(learning_rate=LEARNING_RATE)
model.compile(loss="mse", optimizer=optimizer)
model.summary()

history=model.fit(
    dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience=20)],
)

model.save(MODEL_PATH)

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
