import numpy as np
from ruamel.yaml import YAML
from sklearn.preprocessing import StandardScaler
from urllib.parse import unquote,urlparse
from pathlib import Path
import matplotlib.pyplot as plt

from common.function.function import to_yaml_safe,array_of_array_to_dataset
from common.function.save import save_predict_data
from common.function.model import predict
from common.schema import RnnConfig, SaveConfig
from simulation.configs.schema import SimulationConfig

def calc_fading(simu_cfg: SimulationConfig,seed=42):
    rnd=np.random.RandomState(seed)
    theta = rnd.rand(simu_cfg.l) * 2 * np.pi
    phi = rnd.rand(simu_cfg.l) * 2 * np.pi

    fading_data_list = []
    x = 0.0
    for _ in range(simu_cfg.data_num):
        x += simu_cfg.delta_d
        fading_data = np.sum(
            simu_cfg.r
            * np.exp(1j * (theta + (2 * np.pi / simu_cfg.lambda_0) * x * np.cos(phi)))
        )
        fading_data /= np.sqrt(simu_cfg.l)
        fading_data_list.append(fading_data)

    return np.array(fading_data_list)


def calc_nakagami_rice_fading(simu_cfg: SimulationConfig,seed=42):
    #theta0 = np.random.rand() * 2 * np.pi
    theta0 = 0
    scattered_data_list = calc_fading(simu_cfg,seed)
    scattered_data_list = (
        scattered_data_list
        / np.sqrt(np.mean(np.abs(scattered_data_list) ** 2))
        * np.sqrt(1 / (simu_cfg.k_rice + 1))
    )
    direct_data_list = []
    x = 0
    for _ in range(simu_cfg.data_num):
        x += simu_cfg.delta_d
        direct_data = np.sqrt(simu_cfg.k_rice / (simu_cfg.k_rice + 1.0)) * np.exp(
        #direct_data =  np.exp(
            1j * ((2 * np.pi / simu_cfg.lambda_0) * x + theta0)
        )
        direct_data_list.append(direct_data)
    direct_data_list = np.array(direct_data_list)

    nakagami_rice_data_list = scattered_data_list + direct_data_list

    return nakagami_rice_data_list

'''
### シミュレーション用のデータセット(入力と答え)をdata_set_num分用意する関数
def generate_fading_dataset(
    rnn_cfg:RnnConfig, simu_cfg: SimulationConfig, scaler: StandardScaler | None = None
):
    # fadingデータ作成
    fading_data_arr = []
    for _ in range(simu_cfg.data_set_num):
        fading_data_arr.append(calc_nakagami_rice_fading(simu_cfg))
    fading_data_arr = np.array(fading_data_arr)
    power = np.abs(fading_data_arr) ** 2
    power_db = mw_to_dbm(power)

    # 標準化
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(power_db.reshape(-1, 1))
    data_norm_arr = scaler.transform(power_db.reshape(-1, 1)).reshape(power_db.shape)

    dataset=array_of_array_to_dataset(data_norm_arr,rnn_cfg)
    return dataset, scaler


### シミュレーションデータセットを訓練用、検証用と用意する関数
def load_fading_dataset(simu_cfg: SimulationConfig, rnn_cfg: RnnConfig):
    train_dataset, scaler = generate_fading_dataset(rnn_cfg, simu_cfg)
    val_simu_cfg = simu_cfg.model_copy(update={"data_set_num":simu_cfg.data_set_num//5})
    val_dataset, scaler = generate_fading_dataset(rnn_cfg, val_simu_cfg, scaler)
    return (train_dataset, val_dataset), scaler
'''

### シミュレーション用のデータセット(入力と答え)をdata_set_num分用意する関数
def generate_fading_dataset(
    rnn_cfg:RnnConfig, start,end, scaler: StandardScaler | None = None
):
    from test import csvs
    power_db=csvs[start:end]

    # 標準化
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(power_db.reshape(-1, 1))
    data_norm_arr = scaler.transform(power_db.reshape(-1, 1)).reshape(power_db.shape)

    dataset=array_of_array_to_dataset(data_norm_arr,rnn_cfg)
    return dataset, scaler

### シミュレーションデータセットを訓練用、検証用と用意する関数
def load_fading_dataset(simu_cfg: SimulationConfig, rnn_cfg: RnnConfig):
    train_dataset, scaler = generate_fading_dataset(rnn_cfg, 3,3+simu_cfg.data_set_num)
    val_dataset, scaler = generate_fading_dataset(rnn_cfg, 3+simu_cfg.data_set_num,3+simu_cfg.data_set_num+simu_cfg.data_set_num//5, scaler)
    return (train_dataset, val_dataset), scaler


'''
### 複数のデータセットで予測を行いrmseの平均を算出する 1回目のデータセットだけ詳細な情報を返す
def evaluate_model(
    model,
    scaler: StandardScaler,
    simu_cfg: SimulationConfig,
    rnn_cfg: RnnConfig,
    save_cfg: SaveConfig,
):
    # 中上ライスのデータを取得(kerasモデルに渡せるように加工されていない状態)
    fading_data = calc_nakagami_rice_fading(simu_cfg)
    power = np.abs(fading_data) ** 2
    power_db = 10 * np.log10(power)
    power_db = power_db.reshape(-1, 1)

    # predict関数の中でkerasモデルに渡せるように加工したり正規化などをしている
    # ???create_model関数には加工してからデータを渡すのに、predict関数には加工前のデータを渡してるの変じゃない?
    first_result = predict(
        model,
        power_db,
        scaler,
        rnn_cfg,
        save_cfg.plot_start,
        save_cfg.plot_range,
    )
    
    rmse_sum = np.copy(first_result["rmse_arr"])
    print(first_result["rmse_arr"])
    predict_num = simu_cfg.predicted_dataset_num

    for _ in range(predict_num - 1):  # ↑で一回実行してるのでその分減らす
        fading_data = calc_nakagami_rice_fading(simu_cfg)
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
        )
        rmse_sum += result_i["rmse_arr"]

    rmse_mean_arr = rmse_sum / predict_num
    return first_result, rmse_mean_arr

'''
### 複数のデータセットで予測を行いrmseの平均を算出する 1回目のデータセットだけ詳細な情報を返す
def evaluate_model(
    model,
    scaler: StandardScaler,
    simu_cfg: SimulationConfig,
    rnn_cfg: RnnConfig,
    save_cfg: SaveConfig,
):
    from test import read_csv
    # 中上ライスのデータを取得(kerasモデルに渡せるように加工されていない状態)
    power_db = read_csv(0)

    # predict関数の中でkerasモデルに渡せるように加工したり正規化などをしている
    # ???create_model関数には加工してからデータを渡すのに、predict関数には加工前のデータを渡してるの変じゃない?
    first_result = predict(
        model,
        power_db,
        scaler,
        rnn_cfg,
        save_cfg.plot_start,
        save_cfg.plot_range,
    )
    
    rmse_sum = np.copy(first_result["rmse_arr"])
    print(first_result["rmse_arr"])
    predict_num = simu_cfg.predicted_dataset_num

    for i in range(predict_num - 1):  # ↑で一回実行してるのでその分減らす
        power_db = read_csv(i+1)
        plt.close("all")
        result_i = predict(
            model,
            power_db,
            scaler,
            rnn_cfg,
            save_cfg.plot_start,
            save_cfg.plot_range,
        )
        rmse_sum += result_i["rmse_arr"]

    rmse_mean_arr = rmse_sum / predict_num
    return first_result, rmse_mean_arr

def wrap_save_predict_data(
    run_id,
    first_true_data,
    first_predict_data,
    first_prediict_time,
    first_rmse,
    rmse_mean,
    first_predict_result_fig,
    save_cfg: SaveConfig,
):
    save_predict_data(
        run_id,
        first_true_data,
        first_predict_data,
        first_prediict_time,
        first_rmse,
        first_predict_result_fig,
        save_cfg,
    )
    if save_cfg.use_mlflow:
        import mlflow

        with mlflow.start_run(run_id):
            artifact_dir = mlflow.get_artifact_uri()
            if artifact_dir.startswith("file:"):
                save_path=unquote(urlparse(artifact_dir).path)
                if len(save_path)>=3 and save_path[0]=="/" and save_path[2]==":":
                    save_path=save_path[1:]
            else:
                save_path=artifact_dir

            mlflow.log_metric("rmse_mean", rmse_mean)
    else:
        save_path = save_cfg.save_dir
        
    save_path=Path(save_path)
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)  # インデントの調整
    with open(save_path/"data.yaml", "r") as f:
        data = yaml.load(f)
    data["metrics"]["rmse_mean"] = rmse_mean

    data = to_yaml_safe(data)
    with open(save_path/ "data.yaml", "w") as f:
        yaml.dump(data, f)
