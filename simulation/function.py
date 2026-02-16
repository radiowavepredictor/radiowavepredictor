import numpy as np
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter

from common.utils.func import array_of_array_to_dataset, mw_to_dbm
from common import predict
from common import RnnConfig, SaveConfig
from configs.schema import SimulationConfig


# 反射波のフェージング応答(1波形分)を返す
def make_nlos_fading(simu_cfg: SimulationConfig, rnd: RandomState):
    lm = simu_cfg.l-1 # L-直接波の数=L-1
    theta = rnd.rand(lm) * 2 * np.pi
    phi = rnd.rand(lm) * 2 * np.pi

    h = []
    x = 0.0
    for _ in range(simu_cfg.data_num):
        x += simu_cfg.delta_d
        r_i = 1.0  # (多分)1で固定で良い たぶん
        h_i = np.sum(
            r_i
            * np.exp(1j * (theta + (2 * np.pi / simu_cfg.lambda_0) * x * np.cos(phi)))
        )
        h_i /= np.sqrt(lm)
        h.append(h_i)

    return np.array(h)


# 直接波のフェージング応答(1波形分)を返す
def make_los_fading(simu_cfg: SimulationConfig):
    # theta0 = np.random.rand() * 2 * np.pi
    theta0 = 0  # 多分これでいい 多分
    h = []
    x = 0.0
    for _ in range(simu_cfg.data_num):
        x += simu_cfg.delta_d
        h_i = simu_cfg.r0 * np.exp(1j * ((2 * np.pi / simu_cfg.lambda_0) * x + theta0))
        h.append(h_i)

    return np.array(h)


def make_rice_fading(simu_cfg: SimulationConfig, rnd: RandomState):
    h_nlos = make_nlos_fading(simu_cfg, rnd)
    h_los = make_los_fading(simu_cfg)

    print(h_nlos[:5])
    return h_nlos + h_los


### シミュレーション用のデータセット(入力と答え)をdata_set_num分用意する関数
def make_rice_dataset(
    rnn_cfg: RnnConfig,
    simu_cfg: SimulationConfig,
    rnd: RandomState,
    scaler: StandardScaler | None = None,
):
    # 中上ライスの応答波の配列 [h1,h2,h3…] を作る
    rice_wave_arr = []
    for _ in range(simu_cfg.data_set_num):
        rice_wave_arr.append(make_rice_fading(simu_cfg,rnd))
    rice_wave_arr = np.array(rice_wave_arr)
    power_rice_wave_arr = np.abs(rice_wave_arr) ** 2
    power_db_rice_wave_arr = mw_to_dbm(power_rice_wave_arr)

    # 標準化
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(power_db_rice_wave_arr.reshape(-1, 1))
    data_norm_arr = scaler.transform(power_db_rice_wave_arr.reshape(-1, 1)).reshape(
        power_db_rice_wave_arr.shape
    )

    dataset = array_of_array_to_dataset(data_norm_arr, rnn_cfg)
    return dataset, scaler


### 中上ライスデータセットを訓練用、検証用と用意する
def make_rice_learning_dataset(
    simu_cfg: SimulationConfig, rnn_cfg: RnnConfig, rnd: RandomState
):
    train_dataset, scaler = make_rice_dataset(rnn_cfg, simu_cfg, rnd)
    # 検証用のデータセットをつくるために、データセットの数を訓練用の1/4に゙設定し直す
    val_simu_cfg = simu_cfg.model_copy(
        update={"data_set_num": simu_cfg.data_set_num // 4}
    )
    val_dataset, scaler = make_rice_dataset(rnn_cfg, val_simu_cfg, rnd, scaler)
    return train_dataset, val_dataset, scaler


def predict_multiple_waves(
    model,
    scaler: StandardScaler,
    rnd:RandomState,
    simu_cfg: SimulationConfig,
    rnn_cfg: RnnConfig,
    save_cfg: SaveConfig,
):
    """
    複数のデータセットで予測を行いrmseの平均を算出する
    1回目のデータセットだけ詳細な情報を返す
    """
    # 中上ライスのデータを取得(kerasモデルに渡せるように加工されていない状態)
   
    predict_num = simu_cfg.predicted_dataset_num
    rmse_sum = Counter({})
    for i in range(predict_num):
        fading_data = make_rice_fading(simu_cfg,rnd)
        power = np.abs(fading_data) ** 2
        power_db = 10 * np.log10(power)
        power_db = power_db.reshape(-1, 1)
        plt.close("all")
        result_i = predict(
            model,
            power_db,
            scaler,
            rnn_cfg,
            save_cfg.plot_start,
            save_cfg.plot_range,
            simu_cfg.delta_d,
        )
        if i == 0:
            first_result = result_i
        rmse_sum += Counter(result_i.rmse)

    rmse_mean_dict = {}
    for key, value in rmse_sum.items():
        rmse_mean_dict[f"mean-{key}"] = value / predict_num
    return first_result, rmse_mean_dict