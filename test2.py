import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import joblib
from keras.models import load_model

from common.function.function import predict
from common.schema import RnnConfig
from common.registory import RNNType,OptimizerType
from test import csvs

run_id_in_10="9d2e98406e2c4be691506074be59b35c"
run_id_in_50="7e535f1789bb481783352f70251f8c0d"

out_steps=1
dataset_num=16
plot_start=100
plot_range=50

def search_mlflow(run_id):
    # run_idで作ったmodelを探す
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    #model_path = client.download_artifacts(run_id, "artifacts/model.keras")
    model_path = client.download_artifacts(run_id, "model.keras")
    #scaler_path = client.download_artifacts(run_id,"artifacts/scaler.pkl")
    scaler_path = client.download_artifacts(run_id,"scaler.pkl")
    #true_path=client.download_artifacts(run_id,"artifacts/true.npy")
    true_path=client.download_artifacts(run_id,"true.npy")
    #predicted_path=client.download_artifacts(run_id,"artifacts/predicted.npy")
    predicted_path=client.download_artifacts(run_id,"predicted.npy")

    model = load_model(model_path)
    scaler=joblib.load(scaler_path)
    true=np.load(true_path)
    predict=np.load(predicted_path)
    
    return model,scaler,true,predict
   
model_in_10,scaler,true,predict1=search_mlflow(run_id_in_10)
model_in_50,scaler2,true2,predict2=search_mlflow(run_id_in_50)
print("\n\n")
print("########予測の実行結果########")

simu_data=csvs[1]
print(simu_data)
simu_data_norm = simu_data / np.sqrt(np.mean(np.abs(simu_data) ** 2))
simu_data_norm=simu_data_norm.reshape(-1,1)
result=predict(
    model_in_10,
    simu_data_norm,
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
    100
)
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
    100
)
dt=0.05
plt.close("all")
x_true_data = np.linspace(
    plot_start *dt, (plot_start + plot_range) *dt , plot_range
)
start_10=(10+out_steps-1-plot_start) if plot_start<10+out_steps-1 else 0
x_predict_10 = np.linspace(
    (plot_start +start_10)  *dt,
    (plot_start + plot_range) *dt,
    plot_range - start_10
)
start_50=(50+out_steps-1-plot_start) if plot_start<50+out_steps-1 else 0
x_predict_50=np.linspace(
    (plot_start+start_50)*dt,
    (plot_start+plot_range)*dt,
    plot_range-start_50
)

fig = plt.figure(figsize=(8,3.5))
plt.xlabel("経過時間[s]")
plt.ylabel("受信電力レベル[dBm]")
plt.plot(
    x_true_data,
    simu_data[plot_start : plot_start + plot_range],
    color="black",
    alpha=0.5,
    linewidth=1,
    marker="o",        
    markersize=3,
    markerfacecolor="black",
    markeredgecolor="black",

    label="実測値",
)
plt.plot(
    x_predict_10,
    result["predict_data"][plot_start + start_10 -10-out_steps+1 : plot_start + plot_range-10-out_steps+1,out_steps-1],
    color="tab:green",
    linestyle="-",
    alpha=0.9,
    marker="o",        
    markersize=3,
    markerfacecolor="green",
    markeredgecolor="green",


    linewidth=1,
    label="予測値(入力長-10)",
)

plt.plot(
    x_predict_50,
    result_2["predict_data"][plot_start+start_50-50-out_steps+1 : plot_start + plot_range - 50-out_steps+1,out_steps-1],
    color="tab:red",
    linestyle="-",
    alpha=0.8,
    marker="o",        
    markersize=3,
    markerfacecolor="red",
    markeredgecolor="red",


    linewidth=1,
    label="予測値(入力長-50)",
)
plt.legend()
fig.savefig(f"./simulation/fig/s-n-{dataset_num}-o-{out_steps}.svg", bbox_inches="tight")

print(np.sqrt(np.mean((simu_data[10-out_steps:-out_steps] - simu_data[10:]) ** 2)))
print(f"rmse:: {np.sqrt(np.mean((result_2["predict_data"][plot_start+start_50-50-out_steps+1 : plot_start + plot_range - 50-out_steps+1,out_steps-1]-simu_data[plot_start : plot_start + plot_range])**2))}")
print(result['rmse_arr'])
print(result_2['rmse_arr'])

plt.show()
