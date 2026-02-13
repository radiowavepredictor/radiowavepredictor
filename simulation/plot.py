import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib # importするだけで意味があるので消さない
import joblib
from keras.models import load_model
import time

from test import read_csv
from common import predict
from common import RnnConfig
from common.registory import RNNType,OptimizerType
from common.utils.func import predict_plot_setting

run_id_in_10="2c4e20e0ae104421b294644396c618d6"
run_id_in_50="8be8de27172a4febb0f5c4da50857580"

out_steps=1
dataset_num=16
sampling_rate=0.03925
plot_start=128
plot_range=50

def search_mlflow(run_id):
    # run_idで作ったmodelを探す
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    #model_path = client.download_artifacts(run_id, "model.keras")
    model_path = client.download_artifacts(run_id, "artifacts/model.keras")
    #scaler_path = client.download_artifacts(run_id,"scaler.pkl")
    scaler_path = client.download_artifacts(run_id,"artifacts/scaler.pkl")
    #true_path=client.download_artifacts(run_id,"true_data.npy")
    #predicted_path=client.download_artifacts(run_id,f"predict_data/step-{out_steps}.npy")

    model = load_model(model_path)
    scaler=joblib.load(scaler_path)
    #true=np.load(true_path)
    #predict=np.load(predicted_path)
    
    #return model,scaler,true,predict
    return model,scaler
   
#model_in_10,scaler,true,predict1=search_mlflow(run_id_in_10)
model_in_10,scaler=search_mlflow(run_id_in_10)
#model_in_50,scaler2,true2,predict2=search_mlflow(run_id_in_50)
model_in_50,scaler2=search_mlflow(run_id_in_50)
print("\n\n")
print("########予測の実行結果########")

#simu_data=np.load("power_db.npy")
simu_data=read_csv(0)
print(simu_data)
simu_data=simu_data.reshape(-1,1)

start_10_time=time.time()
result=predict(
    model_in_10,
    simu_data,
    scaler,
    RnnConfig(
        rnn_type=RNNType.SimpleRNN,
        optimizer_type=OptimizerType.Adam,
        in_features=1,
        out_steps_num=out_steps,
        input_len=10,
        hidden_nums=[20],
        batch_size=128,
        epochs=100,
        learning_rate=0.001,
        patience=10
    ),
    0,
    100,
    sampling_rate
)
end_10=time.time()
start_50_time=time.time()
result_2=predict(
    model_in_50,
    simu_data,
    scaler,
    RnnConfig(
        rnn_type=RNNType.SimpleRNN,
        optimizer_type=OptimizerType.Adam,
        in_features=1,
        out_steps_num=out_steps,
        input_len=50,
        hidden_nums=[20],
        batch_size=128,
        epochs=100,
        learning_rate=0.001,
        patience=10
    ),
    0,
    100,
    sampling_rate
)
end_50=time.time()
plt.close("all")
x_arange_true = np.arange(plot_start,plot_start+plot_range)*sampling_rate
# 10サンプル入力の場合
x_arange_10,predict_index_10 = predict_plot_setting(10,sampling_rate,plot_start,plot_range,out_steps)
x_arange_50,predict_index_50 = predict_plot_setting(50,sampling_rate,plot_start,plot_range,out_steps)

fig = plt.figure(figsize=(8,3.5))
plt.xlabel("移動距離[m]")
plt.ylabel("受信電力レベル[dB]")
plt.plot(
    x_arange_true,
    simu_data[plot_start : plot_start + plot_range],
    color="black",
    alpha=0.5,
    linewidth=1.3,
    marker="o",
    markersize=3.6,
    markerfacecolor="black",
    markeredgecolor="black",
    label="実測値",

)
plt.plot(
    x_arange_10,
    result.predict_data[f"step-{out_steps}"][predict_index_10],
    linestyle="--",
    color="green",
    alpha=0.9,
    marker="o",        
    markersize=3.6,
    markerfacecolor="green",
    markeredgecolor="green",
    linewidth=1.4,

    label="予測値(入力長-10)",
)

plt.plot(
    x_arange_50,
    result_2.predict_data[f"step-{out_steps}"][predict_index_50],
    color="tab:red",
    linestyle="--",
    alpha=0.9,
    marker="o",        
    markersize=3.6,
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=1.4,


    label="予測値(入力長-50)",
)
plt.grid(True)
plt.legend()
fig.savefig(f"./fig/v2-s-n-{dataset_num}-o-{out_steps}.svg", bbox_inches="tight")

print(f"データをスライドしただけのRMSE{np.sqrt(np.mean((simu_data[10-out_steps:-out_steps] - simu_data[10:]) ** 2))}")
#print(f"rmse:: {np.sqrt(np.mean((result_2["predict_data"][plot_start+start_50-50-out_steps+1 : plot_start + plot_range - 50-out_steps+1,out_steps-1]-simu_data[plot_start : plot_start + plot_range])**2))}")
print(result.rmse)
print(result_2.rmse)
print(f"10の予測時間{end_10-start_10_time:.6f}秒")
print(f"50の予測時間{end_50-start_50_time:.6f}秒")

plt.show()