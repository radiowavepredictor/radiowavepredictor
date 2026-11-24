import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import japanize_matplotlib # importするだけで意味があるので消さない
import yaml
from pathlib import Path

from common.utils.func import predict_plot_setting


run_id_in_50="447e597a216843448b0b58e1e7f89b43"
run_id_in_100="6bc84306a0544c3ebec31ef9eebfa15f"

# ============================================================
# 横軸の設定 (3GPP版 plot.py と同じ値にする)
# ============================================================
# 表示範囲をサンプル番号ではなく移動距離[m]で決めるので、
# delta_d が違うシミュレーション同士でも同じ横軸になる
PLOT_START_M = 3.0  # 表示開始位置[m]
PLOT_RANGE_M = 1.0  # 表示する長さ[m]
X_TICK_M = 0.2  # 横軸の目盛り間隔[m]

# ============================================================
# grid_cfg.yaml を読み込む (MLflowに設定が残っていないときの予備)
# ============================================================
# plot.py が置かれているフォルダ
BASE_DIR = Path(__file__).resolve().parent

# simulation/configs/grid_cfg.yaml
CONFIG_PATH = BASE_DIR / "configs" / "grid_cfg.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg_yaml = yaml.safe_load(f)

simulation_cfg_yaml = cfg_yaml["params"]["simulation"]
model_cfg_yaml = cfg_yaml["params"]["model"]


def search_mlflow(run_id, input_len_default):
    """保存済みの元値・予測値と、その run を学習したときの設定を取得する"""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # yaml は後から書き換わることがあるので、run に保存された値を優先する
    params = client.get_run(run_id).data.params
    run_cfg = {
        "delta_d": float(params.get("delta_d", simulation_cfg_yaml["delta_d"])),
        "input_len": int(params.get("input_len", input_len_default)),
        "out_steps": int(params.get("out_steps_num", model_cfg_yaml["out_steps_num"])),
        "data_set_num": int(params.get("data_set_num", simulation_cfg_yaml["data_set_num"])),
    }

    true_path = client.download_artifacts(
        run_id,
        "true_data.npy"
    )

    predicted_path = client.download_artifacts(
        run_id,
        f"predict_data/step-{run_cfg['out_steps']}.npy"
    )

    true_data = np.load(true_path).ravel()
    predict_data = np.load(predicted_path).ravel()

    return true_data, predict_data, run_cfg


def cut_plot_window(true_data, predict_data, run_cfg):
    """
    PLOT_START_M から PLOT_RANGE_M [m] の範囲を切り出す
    戻り値の横軸はすべて移動距離[m]
    """
    delta_d = run_cfg["delta_d"]
    # 表示範囲の内側に収まるサンプルだけを使う
    plot_start = int(np.ceil(PLOT_START_M / delta_d - 1e-9))
    plot_end = int(np.floor((PLOT_START_M + PLOT_RANGE_M) / delta_d + 1e-9)) + 1
    plot_range = plot_end - plot_start

    if plot_start + plot_range > len(true_data):
        raise ValueError(
            f"表示範囲 {PLOT_START_M}~{PLOT_START_M + PLOT_RANGE_M}m が"
            f"波形の長さ {len(true_data) * delta_d:.2f}m を超えています"
        )

    x_true = np.arange(plot_start, plot_start + plot_range)

    # 予測値は input_len 点目以降にしか無いので、予測値側の位置を求める
    x_pred, predict_index = predict_plot_setting(
        run_cfg["input_len"],
        delta_d,
        plot_start,
        plot_range,
        run_cfg["out_steps"],
    )
    if len(x_pred) == 0:
        print(
            f"※ 入力長{run_cfg['input_len']}の予測は "
            f"{(run_cfg['input_len'] + run_cfg['out_steps'] - 1) * delta_d:.2f}m から始まるため、"
            f"表示範囲に予測値がありません (PLOT_START_M を大きくしてください)"
        )
        y_pred = np.array([])
        rmse = np.nan
    else:
        y_pred = predict_data[predict_index]
        rmse = np.sqrt(np.mean((true_data[x_pred] - y_pred) ** 2))

    return {
        "x_true": x_true * delta_d,
        "y_true": true_data[x_true],
        "x_pred": x_pred * delta_d,
        "y_pred": y_pred,
        "rmse": rmse,
    }


INPUT_LENS = model_cfg_yaml["input_len"]

true_50, predict_50, cfg_50 = search_mlflow(run_id_in_50, INPUT_LENS[0])
true_100, predict_100, cfg_100 = search_mlflow(run_id_in_100, INPUT_LENS[1])

if cfg_50["delta_d"] != cfg_100["delta_d"]:
    print(f"※ 2つの run の delta_d が違います ({cfg_50['delta_d']} / {cfg_100['delta_d']})")

INPUT_LEN_50 = cfg_50["input_len"]
INPUT_LEN_100 = cfg_100["input_len"]
OUT_STEPS = cfg_50["out_steps"]
DATA_SET_NUM = cfg_50["data_set_num"]

win_50 = cut_plot_window(true_50, predict_50, cfg_50)
win_100 = cut_plot_window(true_100, predict_100, cfg_100)

print("\n\n")
print("########予測の実行結果########")
print(f"delta_d: {cfg_50['delta_d']} m, 表示範囲: {PLOT_START_M}~{PLOT_START_M + PLOT_RANGE_M} m")
plt.close("all")

fig = plt.figure(figsize=(8,3.5))
plt.xlabel("移動距離[m]")
plt.ylabel("受信電力レベル[dB]")
plt.plot(
    win_50["x_true"],
    win_50["y_true"],
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
    win_50["x_pred"],
    win_50["y_pred"],
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
    win_100["x_pred"],
    win_100["y_pred"],
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

# 横軸の範囲と目盛りを固定する (3GPP版と同じ)
plt.xlim(PLOT_START_M, PLOT_START_M + PLOT_RANGE_M)
plt.gca().xaxis.set_major_locator(MultipleLocator(X_TICK_M))
plt.grid(True)
plt.legend()

Path("./fig").mkdir(exist_ok=True)
fig.savefig(f"./fig/v2-s-n-{DATA_SET_NUM}-o-{OUT_STEPS}.svg", bbox_inches="tight")

# RMSE (表示範囲内で、元値と予測値の位置を合わせて計算)
print(f"input_len={INPUT_LEN_50} RMSE: {win_50['rmse']}")
print(f"input_len={INPUT_LEN_100} RMSE: {win_100['rmse']}")

plt.show()