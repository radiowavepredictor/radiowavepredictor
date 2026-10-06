import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib # importするだけで意味があるので消さない
import yaml

from common.utils.func import predict_plot_setting


run_id_in_50="d85e82172a6c44f098d6c0cb2b21cf87"
run_id_in_100="accea4c18ccf48ffbfbf3b57323e15a5"

# ============================================================
# grid_cfg.yaml を読み込む
# ============================================================
from pathlib import Path

# plot.py が置かれているフォルダ
BASE_DIR = Path(__file__).resolve().parent

# simulation/configs/grid_cfg.yaml
CONFIG_PATH = BASE_DIR / "configs" / "grid_cfg.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg_yaml = yaml.safe_load(f)

simulation_cfg_yaml = cfg_yaml["params"]["simulation"]
model_cfg_yaml = cfg_yaml["params"]["model"]
save_cfg_yaml = cfg_yaml["params"]["save"]


# simulation
DATA_SET_NUM = simulation_cfg_yaml["data_set_num"]
SAMPLING_RATE = simulation_cfg_yaml["delta_d"]

# model
INPUT_LENS = model_cfg_yaml["input_len"]
OUT_STEPS = model_cfg_yaml["out_steps_num"]

# save
PLOT_START = save_cfg_yaml["plot_start"]
PLOT_RANGE = save_cfg_yaml["plot_range"]

def search_mlflow(run_id):
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    true_path = client.download_artifacts(
        run_id,
        "true_data.npy"
    )

    predicted_path = client.download_artifacts(
        run_id,
        f"predict_data/step-{OUT_STEPS}.npy"
    )

    true_data = np.load(true_path)
    predict_data = np.load(predicted_path)

    return true_data, predict_data
   
true_50, predict_50 = search_mlflow(run_id_in_50)
true_100, predict_100 = search_mlflow(run_id_in_100)
print("\n\n")
print("########予測の実行結果########")
plt.close("all")
#x_arange_true = np.arange(plot_start,plot_start+plot_range)*sampling_rate
# 10サンプル入力の場合
#x_arange_10 = x_arange_ture ,predict_index_10 = predict_plot_setting(50,sampling_rate,plot_start,plot_range,out_steps)
#x_arange_50 = x_arange_ture ,predict_index_50 = predict_plot_setting(100,sampling_rate,plot_start,plot_range,out_steps)

# 実測値の横軸
x_arange_true = (
    np.arange(
        PLOT_START,
        PLOT_START + PLOT_RANGE
    )
    * SAMPLING_RATE
)

# 予測データの切り出し位置だけ取得
INPUT_LEN_50 = INPUT_LENS[0]
INPUT_LEN_100 = INPUT_LENS[1]

_, predict_index_50 = predict_plot_setting(
    INPUT_LEN_50,
    SAMPLING_RATE,
    PLOT_START,
    PLOT_RANGE,
    OUT_STEPS
)

_, predict_index_100 = predict_plot_setting(
    INPUT_LEN_100,
    SAMPLING_RATE,
    PLOT_START,
    PLOT_RANGE,
    OUT_STEPS
)

# 実測値と予測値で同じ横軸を使用
x_arange_50 = x_arange_true
x_arange_100 = x_arange_true

fig = plt.figure(figsize=(8,3.5))
plt.xlabel("移動距離[m]")
plt.ylabel("受信電力レベル[dB]")
plt.plot(
    x_arange_true,
    true_50[
        PLOT_START:PLOT_START + PLOT_RANGE
    ],
    color="black",
    alpha=0.5,
    linewidth=1.3,
    marker="o",
    markersize=3.6,
    markerfacecolor="black",
    markeredgecolor="black",
    label="元値",
)

plt.plot(
    x_arange_50,
    predict_50[predict_index_50],
    linestyle="--",
    color="green",
    alpha=0.9,
    marker="o",
    markersize=3.6,
    markerfacecolor="green",
    markeredgecolor="green",
    linewidth=1.4,
    label=f"予測値(入力長-{INPUT_LEN_50})",
)

plt.plot(
    x_arange_100,
    predict_100[predict_index_100],
    color="tab:red",
    linestyle="--",
    alpha=0.9,
    marker="o",
    markersize=3.6,
    markerfacecolor="red",
    markeredgecolor="red",
    linewidth=1.4,
    label=f"予測値(入力長-{INPUT_LEN_100})",
)
plt.grid(True)
plt.legend()
fig.savefig(f"./fig/v2-s-n-{DATA_SET_NUM}-o-{OUT_STEPS}.svg", bbox_inches="tight")

true_plot_50 = true_50[
    PLOT_START:PLOT_START + PLOT_RANGE
]

pred_plot_50 = predict_50[predict_index_50]

true_plot_100 = true_100[
    PLOT_START:PLOT_START + PLOT_RANGE
]

pred_plot_100 = predict_100[predict_index_100]


rmse_50 = np.sqrt(
    np.mean((true_plot_50 - pred_plot_50) ** 2)
)

rmse_100 = np.sqrt(
    np.mean((true_plot_100 - pred_plot_100) ** 2)
)

print(f"input_len={INPUT_LEN_50} RMSE: {rmse_50}")
print(f"input_len={INPUT_LEN_100} RMSE: {rmse_100}")

plt.show()