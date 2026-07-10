import numpy as np
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter

from common.utils.func import array_of_array_to_dataset, mw_to_dbm
from common import predict
from common import RnnConfig, SaveConfig
from configs.schema import SimulationConfig
from configs.fading_schema import RiceConfig,LosConfig,NLosConfig


# 反射波のフェージング応答(1波形分)を返す
def make_nlos_fading(nlos_cfg:NLosConfig , rnd: RandomState):
    lm = nlos_cfg.l-1 # L-直接波の数=L-1
    theta = rnd.rand(lm) * 2 * np.pi
    phi = rnd.rand(lm) * 2 * np.pi

    h = []
    x = 0.0
    for _ in range(nlos_cfg.data_num):
        x += nlos_cfg.delta_d
        r_i = 1.0  # (多分)1で固定で良い たぶん
        h_i = np.sum(
            r_i
            * np.exp(1j * (theta + (2 * np.pi / nlos_cfg.lambda_0) * x * np.cos(phi)))
        )
        h_i /= np.sqrt(lm)
        h.append(h_i)

    return np.array(h)
    
# 直接波のフェージング応答(1波形分)を返す
def make_los_fading(los_cfg:LosConfig):
    # theta0 = np.random.rand() * 2 * np.pi
    theta0 = 0  # 多分これでいい 多分
    h = []
    x = 0.0
    for _ in range(los_cfg.data_num):
        x += los_cfg.delta_d
        h_i = los_cfg.r0 * np.exp(1j * ((2 * np.pi / los_cfg.lambda_0) * x + theta0))
        
        h.append(h_i)

    return np.array(h)

def make_rice_fading(rice_cfg: RiceConfig, rnd: RandomState):
    h_nlos = make_nlos_fading(rice_cfg, rnd)
    h_los = make_los_fading(rice_cfg)

    return h_nlos + h_los

#シャドウイング波形を生成
def make_shadowing(simu_cfg, rnd):
    data_num=simu_cfg.data_num
    sigma_shadow=simu_cfg.sigma_shadow
    d_c=simu_cfg.d_c 
    
    a=0.5**(simu_cfg.delta_d/d_c)
    shadow=np.zeros(data_num)
    b=np.zeros(data_num)

    b[0]=rnd.normal(0,sigma_shadow)
    shadow[0]=b[0]

    for n in range(1,data_num):
        b[n]=rnd.normal(0,sigma_shadow)

        shadow[n]=a*shadow[n-1]+np.sqrt(1-a**2)*b[n]
        
    return shadow
 #マルチパスとシャドウイングの統合
def make_rice_shadowing(simu_cfg, rnd):
    fading = make_rice_fading(simu_cfg, rnd)
    power = np.abs(fading)**2
    power_db = mw_to_dbm(power)
    shadow = make_shadowing(simu_cfg, rnd)
    receive_power = power_db + shadow
    return receive_power
    



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

    plt.figure(figsize=(10,4))
    plt.plot(rice_wave_arr[0])
    plt.grid(True)
    plt.show()

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


###シミュレーション用のデータセット(シャドウイング版)
def make_shadow_dataset(
    rnn_cfg: RnnConfig,
    simu_cfg: SimulationConfig,
    rnd: RandomState,
    scaler: StandardScaler | None = None,
):
    # シャドウイングの応答波の配列 [h1,h2,h3…] を作る
    shadow_wave_arr = []
    for _ in range(simu_cfg.data_set_num):
        shadow_wave_arr.append(make_shadowing(simu_cfg,rnd))
    shadow_wave_arr = np.array(shadow_wave_arr)

    plt.figure(figsize=(10,4))
    plt.plot(shadow_wave_arr[0])
    plt.grid(True)
    plt.show()

   
    # 標準化
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(shadow_wave_arr.reshape(-1, 1))
    data_norm_arr = scaler.transform(shadow_wave_arr.reshape(-1, 1)).reshape(
        shadow_wave_arr.shape
    )

    dataset = array_of_array_to_dataset(data_norm_arr, rnn_cfg)
    return dataset, scaler


def make_shadow_learning_dataset(
        simu_cfg,rnn_cfg,rnd
):
    train_dataset, scaler = make_shadow_dataset(rnn_cfg, simu_cfg, rnd)
    # 検証用のデータセットをつくるために、データセットの数を訓練用の1/4に゙設定し直す
    val_simu_cfg = simu_cfg.model_copy(
        update={"data_set_num": simu_cfg.data_set_num // 4}
    )
    val_dataset, scaler = make_shadow_dataset(rnn_cfg, val_simu_cfg, rnd, scaler)
    return train_dataset, val_dataset, scaler
    
###シミュレーション用のデータセット(マルチパス＋シャドウイング版)
def make_rice_shadow_dataset(
    rnn_cfg: RnnConfig,
    simu_cfg: SimulationConfig,
    rnd: RandomState,
    scaler: StandardScaler | None = None,
):
    # シャドウイングの応答波の配列 [h1,h2,h3…] を作る
    rice_shadow_wave_arr = []
    for _ in range(simu_cfg.data_set_num):
        rice_shadow_wave_arr.append(make_rice_shadowing(simu_cfg,rnd))
    rice_shadow_wave_arr = np.array(rice_shadow_wave_arr)

    plt.figure(figsize=(10,4))
    plt.plot(rice_shadow_wave_arr[0])
    plt.grid(True)
    plt.show()

   
    # 標準化
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(rice_shadow_wave_arr.reshape(-1, 1))
    data_norm_arr = scaler.transform(rice_shadow_wave_arr.reshape(-1, 1)).reshape(
        rice_shadow_wave_arr.shape
    )

    dataset = array_of_array_to_dataset(data_norm_arr, rnn_cfg)
    return dataset, scaler


def make_rice_shadow_learning_dataset(
        simu_cfg,rnn_cfg,rnd
):
    train_dataset, scaler = make_rice_shadow_dataset(rnn_cfg, simu_cfg, rnd)
    # 検証用のデータセットをつくるために、データセットの数を訓練用の1/4に゙設定し直す
    val_simu_cfg = simu_cfg.model_copy(
        update={"data_set_num": simu_cfg.data_set_num // 4}
    )
    val_dataset, scaler = make_rice_shadow_dataset(rnn_cfg, val_simu_cfg, rnd, scaler)
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
    #shadow_db=make_shadowing(simu_cfg,rnd)
    predict_num = simu_cfg.predicted_dataset_num
    rmse_sum = Counter({})
    for i in range(predict_num):

        #マルチパスフェージングのみ
        #fading_data = make_rice_fading(simu_cfg,rnd)
        #power = np.abs(fading_data) ** 2
        #power_db = 10 * np.log10(power)
        #power_db = power_db.reshape(-1, 1)
        #rice_data=make_rice_fading(simu_cfg,rnd)
        #input_data=rice_data.reshape(-1,1)

        #シャドウイングのみ
        shadow_data=make_shadowing(simu_cfg,rnd)
        input_data=shadow_data.reshape(-1,1)


        #マルチパスとシャドウイング統合
        #rice_shadow_data=make_rice_shadowing(simu_cfg,rnd)
        #input_data=rice_shadow_data.reshape(-1,1)

        plt.close("all")
        result_i = predict(
            model,
            input_data,
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




