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



def make_path_loss(simu_cfg,):
    fc = simu_cfg.f / 1e9      
    c = simu_cfg.c

    h_bs = 25.0                # BS基地局の高さ[m]
    h_ut = 1.5                 # UE受信局の高さ[m]

    # 2次元距離
    # d2d = np.linspace(10, 100, simu_cfg.data_num)

    d2d = np.arange(simu_cfg.data_num) * simu_cfg.delta_d+10

    # 3次元距離
    d3d = np.sqrt(d2d**2 + (h_bs-h_ut)**2)

    # ブレークポイント
    d_bp = 4 * (h_bs - 1) * (h_ut - 1) * fc * 1e9 / c

    # LOS
    pl_los = np.zeros_like(d3d)
    a = d2d <= d_bp
    pl_los[a] = (
        28+22*np.log10(d3d[a])+20*np.log10(fc)
    )
    a = d2d > d_bp
    pl_los[a] = (
        28+40*np.log10(d3d[a])+20*np.log10(fc)-9*np.log10(d_bp**2+(h_bs-h_ut)**2)
    )

    # NLOS
    pl_nlos = (
        13.54+39.08*np.log10(d3d)+20*np.log10(fc)-0.6*(h_ut-1.5)
    )
    pl_nlos = np.maximum(pl_los, pl_nlos)

    # LOS確率
    p_los = np.zeros_like(d2d)

    a = d2d <= 18
    p_los[a] = 1.0

    a = d2d > 18
    p_los[a] = (
        18/d2d[a]+ np.exp(-d2d[a]/63)* (1-18/d2d[a])
    )

    # LOS/NLOS選択
   # rand = np.random.rand(len(d2d))
   # path_loss = np.where(rand < p_los, pl_los, pl_nlos)

    if np.random.rand() < np.mean(p_los):
        path_loss = pl_los
    else:
        path_loss = pl_nlos


    return path_loss

def make_rice_shadow_pathloss(simu_cfg, rnd):
    fading = make_rice_fading(simu_cfg, rnd)
    power_db = mw_to_dbm(np.abs(fading) ** 2)
    shadow = make_shadowing(simu_cfg, rnd)
    path_loss = make_path_loss(simu_cfg)
    receive_power = power_db + shadow - path_loss

    return receive_power

def make_rice_shadow_pathloss_dataset(
    rnn_cfg: RnnConfig,
    simu_cfg: SimulationConfig,
    rnd: RandomState,
    scaler: StandardScaler | None = None,
):
    # マルチパス＋シャドウイング＋距離特性の波形を作成
    rice_shadow_pathloss_wave_arr = []

    for _ in range(simu_cfg.data_set_num):
        rice_shadow_pathloss_wave_arr.append(
            make_rice_shadow_pathloss(simu_cfg, rnd)
        )

    rice_shadow_pathloss_wave_arr = np.array(rice_shadow_pathloss_wave_arr)

    dist = np.arange(simu_cfg.data_num) * simu_cfg.delta_d+10

    plt.figure(figsize=(10, 4))
    plt.plot(dist,rice_shadow_pathloss_wave_arr[0])
    plt.title("training arr")
    plt.grid(True)
    plt.show()

    # 標準化
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(rice_shadow_pathloss_wave_arr.reshape(-1, 1))

    data_norm_arr = scaler.transform(
        rice_shadow_pathloss_wave_arr.reshape(-1, 1)
    ).reshape(rice_shadow_pathloss_wave_arr.shape)

    dataset = array_of_array_to_dataset(data_norm_arr, rnn_cfg)

    return dataset, scaler

def make_rice_shadow_pathloss_learning_dataset(
    simu_cfg: SimulationConfig,
    rnn_cfg: RnnConfig,
    rnd: RandomState,
):
    train_dataset, scaler = make_rice_shadow_pathloss_dataset(
        rnn_cfg,
        simu_cfg,
        rnd,
    )

    val_simu_cfg = simu_cfg.model_copy(
        update={"data_set_num": simu_cfg.data_set_num // 4}
    )

    val_dataset, scaler = make_rice_shadow_pathloss_dataset(
        rnn_cfg,
        val_simu_cfg,
        rnd,
        scaler,
    )

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
        #shadow_data=make_shadowing(simu_cfg,rnd)
        #input_data=shadow_data.reshape(-1,1)

        # パスロスのみ
        #pathloss_data = make_path_loss(simu_cfg)
        #input_data = pathloss_data.reshape(-1, 1)

        #マルチパスとシャドウイング統合
        #rice_shadow_data=make_rice_shadowing(simu_cfg,rnd)
        #input_data=rice_shadow_data.reshape(-1,1)

        #マルチパス、シャドウイング、距離特性
        rice_shadow_pathloss_data = make_rice_shadow_pathloss(simu_cfg, rnd)
        input_data = rice_shadow_pathloss_data.reshape(-1, 1)
    
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




