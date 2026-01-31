import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import joblib
from keras.models import load_model
import time
from pathlib import Path

from common.function.model import predict
from common.schema import RnnConfig
from common.registory import RNNType,OptimizerType

run_id_in_10="b31f5cc48ac147c0844890b94d145e55"
run_id_in_50="a80d25f52e8d406f8ab5c85fc7794a87"

cource=16

out_steps=5
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

csv_path= Path(".")/"measurement"/"result"/f"WAVE{cource:04d}"/f"result_nt-001.csv" 
data_csv = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
measure_data = data_csv.values.astype(np.float64) # csv用のデータ構造からnumpy配列に変換

in_10_start=time.time()
result=predict(
    model_in_10,
    measure_data,
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
in_10_end=time.time()
in_50_start=time.time()
result_2=predict(
    model_in_50,
    measure_data,
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
in_50_end=time.time()
plt.close("all")

dt=0.05

x_plot = np.arange(plot_start*dt,(plot_start+plot_range)*dt,dt)
# 10サンプル入力の場合
start_10 = (10 + out_steps - 1 - plot_start) if plot_start < 10 + out_steps - 1 else 0
x_predict_10 = np.arange(plot_start + start_10, plot_start + plot_range) * dt

# 50サンプル入力の場合
start_50 = (50 + out_steps - 1 - plot_start) if plot_start < 50 + out_steps - 1 else 0
x_predict_50 = np.arange(plot_start + start_50, plot_start + plot_range) * dt
fig = plt.figure(figsize=(8,3.5))
plt.rcParams["font.size"] = 10.5
plt.xlabel("経過時間[s]")
plt.ylabel("受信電力レベル[dBm]")
plt.plot(
    x_plot,
    measure_data[plot_start : plot_start + plot_range],
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
    x_predict_10,
    result["predict_data"][plot_start + start_10 -10-out_steps+1 : plot_start + plot_range-10-out_steps+1,out_steps-1],
    color="tab:green",
    linestyle="--",
    alpha=0.9,
    marker="o",        
    markersize=3.6,
    markerfacecolor="green",
    markeredgecolor="green",


    linewidth=1.4,
    label="予測値(入力長-10)",
)

plt.plot(
    x_predict_50,
    result_2["predict_data"][plot_start+start_50-50-out_steps+1 : plot_start + plot_range - 50-out_steps+1,out_steps-1],
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
fig.savefig(f"./fig/m-c-{cource}-o-{out_steps}.svg", bbox_inches="tight")

print(f"スライドしたRMSE{np.sqrt(np.mean((measure_data[10-out_steps:-out_steps] - measure_data[10:]) ** 2))}")
#print(f"rmse:: {np.sqrt(np.mean((result_2["predict_data"][plot_start+start_50-50-out_steps+1 : plot_start + plot_range - 50-out_steps+1,out_steps-1]-measure_data[plot_start : plot_start + plot_range])**2))}")
print(result['rmse_arr'])
print(result_2['rmse_arr'])
print(f"10の予測時間{in_10_end-in_10_start:.6f}秒")
print(f"50の予測時間{in_50_end-in_50_start:.6f}秒")


plt.show()
'''
plt.close("all")
fig = plt.figure(figsize=(8,3.5))
plt.rcParams["font.size"] = 10.5
plt.xlabel("経過時間[s]")
plt.ylabel("受信電力レベル[dBm]")
plt.plot(
    x_plot,
    measure_data[plot_start : plot_start + plot_range],
    color="black",
    alpha=0.5,
    linewidth=1,
    label="実測値",
)
plt.plot(
    x_predict_10,
    measure_data[plot_start-1:plot_start+plot_range-1],
    color="tab:green",
    linestyle="-",
    alpha=0.9,


    linewidth=1,
    label="スライド波形-1ステップ",
)
plt.plot(
    x_predict_10,
    measure_data[plot_start-3:plot_start+plot_range-3],
    color="tab:red",
    linestyle="-",
    alpha=0.9,

    linewidth=1,
    label="スライド波形-3ステップ",
)
plt.plot(
    x_predict_10,
    measure_data[plot_start-5:plot_start+plot_range-5],
    color="tab:blue",
    linestyle="-",
    alpha=0.9,


    linewidth=1,
    label="スライド波形-5ステップ",
)
plt.grid(True)
plt.legend()
fig.savefig(f"./fig.svg", bbox_inches="tight")

plt.show()
'''