import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import timeseries_dataset_from_array
import mlflow
from mlflow.tracking import MlflowClient

from simulation_func.setting import *
from simulation_func.simu_func import *
from func import *

with open("./simulation_func/run_id.txt", "r") as f:
    run_id = f.readline().strip()

client = MlflowClient()
model_path = client.download_artifacts(run_id,MODEL_NAME)   
model = load_model(model_path)
    
fading_data=calc_nakagami_rice_fading()
power = np.abs(fading_data) ** 2
power_db = 10 * np.log10(power)
normalized_data = normalize(power_db)

x=timeseries_dataset_from_array(
    normalized_data,
    targets=None,
    sequence_length=INPUT_LEN,
    batch_size=1,
    shuffle=None
)

predicted = model.predict(x)
true_data=power_db
denormalized_predicted = denormalize(predicted,true_data)
reshape_denormalied_predeicted = np.array(denormalized_predicted).reshape(len(denormalized_predicted))
rmse=np.sqrt(np.mean((reshape_denormalied_predeicted[:-1]-true_data[INPUT_LEN:])**2))
print(rmse)

# ここで使うデータは0.05ミリ秒毎にサンプリングされている
# plotするときに単位を秒にするための準備
x_true_data=np.linspace(PLOT_START/20,(PLOT_START+PLOT_RANGE)/20,PLOT_RANGE)
x_predict=np.linspace((PLOT_START+INPUT_LEN)/20,(PLOT_START+PLOT_RANGE)/20,PLOT_RANGE-INPUT_LEN)

plt.figure()
plt.xlabel("Time[s]")
plt.ylabel("ReceivedPower[dBm]")
plt.plot(x_true_data,true_data[PLOT_START:PLOT_START+PLOT_RANGE],color="r",alpha=0.5,label="true_data")
plt.plot(x_predict, denormalized_predicted[PLOT_START:PLOT_START+PLOT_RANGE-INPUT_LEN], color="g", label="predict_data")
plt.legend()

np.save("true.npy", denormalized_predicted)
np.save("predicted.npy", denormalized_predicted)

with mlflow.start_run(run_id):
    mlflow.log_metric("rmse",rmse)
    mlflow.log_artifact("true.npy")
    mlflow.log_artifact("predicted.npy")
    mlflow.log_figure(plt.gcf(), "predict_results.png") 
    
plt.show()