import os
import matplotlib.pyplot as plt
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import mlflow
import time
from datetime import datetime

from simulation_func.simu_func import *

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
model.compile(loss="mse", optimizer=USE_OPTIMIZER(learning_rate=LEARNING_RATE))
model.summary()

start_time=time.time()
history=model.fit(
    dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience=10)],
)

end_time=time.time()
training_time=end_time-start_time


########## グラフ表示とデータ保存 ############
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()

run_name=datetime.now().strftime("%m_%d_%H_%M")

mlflow.set_experiment("simulation")
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_param("L",L)
    mlflow.log_param("Delta_D",DELTA_D)
    mlflow.log_param("Data_Num",DATA_NUM)
    mlflow.log_param("Data_Set_Num",DATA_SET_NUM)
    mlflow.log_param("K_Rice",K_RICE)

    mlflow.log_param("Input", INPUT_LEN)
    mlflow.log_param("Layers",len(HIDDEN_NUMS))
    mlflow.log_param("Units",HIDDEN_NUMS)
    mlflow.log_param("Batch", BATCH_SIZE)
    mlflow.log_param("Learning_Rate", LEARNING_RATE)
    mlflow.log_param("Optimizer", USE_OPTIMIZER.__name__)
    mlflow.log_param("RNN_Name",USE_RNN_LAYER.__name__)
    mlflow.log_param("Epochs", EPOCHS)

    mlflow.log_figure(plt.gcf(), "loss_curve.png") 
    mlflow.log_metric("training_time",training_time)
    artifact_dir = mlflow.get_artifact_uri()
    model_path = os.path.join(artifact_dir.replace("file:",""), MODEL_NAME)
    model.save(model_path)
    run_id=run.info.run_id
    with open("./simulation_func/run_id.txt","w") as f:
        f.write(run_id)
print("\n\n")
print("##############################")
print("experiment_id",run.info.experiment_id)
print("run_id",run_id)
print(f"実行時間:{training_time:.2f}秒")
print("##############################")
plt.show()