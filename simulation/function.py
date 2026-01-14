import os
import numpy as np
from ruamel.yaml import YAML
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array

from common.function import mw_to_dbm, predict, save_predict_data,to_yaml_safe
from common.schema import RnnConfig, SaveConfig
from simulation.configs.schema import SimulationConfig


def calc_fading(simu_cfg: SimulationConfig):
    theta = np.random.rand(simu_cfg.l) * 2 * np.pi
    phi = np.random.rand(simu_cfg.l) * 2 * np.pi

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


def calc_nakagami_rice_fading(simu_cfg: SimulationConfig):
    theta0 = np.random.rand() * 2 * np.pi
    scattered_data_list = calc_fading(simu_cfg)
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
            1j * ((2 * np.pi / simu_cfg.lambda_0) * x + theta0)
        )
        direct_data_list.append(direct_data)
    direct_data_list = np.array(direct_data_list)

    nakagami_rice_data_list = scattered_data_list + direct_data_list

    return nakagami_rice_data_list


### シミュレーション用のデータセット(入力と答え)をdata_set_num分用意する関数
def generate_fading_dataset_list(
    input_len, simu_cfg: SimulationConfig, scaler: StandardScaler | None = None
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

    dataset_arr = []
    for data_norm in data_norm_arr:
        # datasetの中身はtensorflow特有のオブジェクトで入力と出力(入力に対する答え)のセットが入っている
        dataset_i = timeseries_dataset_from_array(
            data=data_norm,
            targets=data_norm[input_len:],
            sequence_length=input_len,
            batch_size=None,  # type:ignore[arg-type]
            shuffle=False,
        )
        dataset_arr.append(dataset_i)
    # dataset_arrの中身をすべてつなげる
    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)
    return dataset, scaler


### シミュレーションデータセットを訓練用、検証用と用意する関数
def load_fading_dataset(simu_cfg: SimulationConfig, rnn_cfg: RnnConfig):
    train_dataset, scaler = generate_fading_dataset_list(rnn_cfg.input_len, simu_cfg)
    train_dataset = (
        train_dataset.shuffle(buffer_size=10000)
        .batch(rnn_cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_simu_cfg = simu_cfg.model_copy(update={"data_set_num":simu_cfg.data_set_num})
    val_dataset, scaler = generate_fading_dataset_list(
        rnn_cfg.input_len, val_simu_cfg, scaler
    )
    val_dataset = val_dataset.batch(rnn_cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return (train_dataset, val_dataset), scaler


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
        rnn_cfg.input_len,
        save_cfg.plot_start,
        save_cfg.plot_range,
    )

    rmse_sum = first_result["rmse"]
    predict_num = simu_cfg.predicted_dataset_num

    for _ in range(predict_num - 1):  # ↑で一回実行してるのでその分減らす
        fading_data = calc_nakagami_rice_fading(simu_cfg)
        power = np.abs(fading_data) ** 2
        power_db = 10 * np.log10(power)
        power_db = power_db.reshape(-1, 1)
        result_i = predict(
            model,
            power_db,
            scaler,
            rnn_cfg.input_len,
            save_cfg.plot_start,
            save_cfg.plot_range,
        )
        rmse_sum += result_i["rmse"]

    rmse_mean = rmse_sum / predict_num
    return first_result, rmse_mean


def wrap_save_predict_data(
    run_id,
    first_true_data,
    first_predict_data,
    first_rmse,
    rmse_mean,
    first_predict_result_fig,
    save_cfg: SaveConfig,
):
    save_predict_data(
        run_id,
        first_true_data,
        first_predict_data,
        first_rmse,
        first_predict_result_fig,
        save_cfg,
    )
    if save_cfg.use_mlflow:
        import mlflow

        with mlflow.start_run(run_id):
            artifact_dir = mlflow.get_artifact_uri()
            save_path = artifact_dir.replace("file:", "")
            mlflow.log_metric("rmse_mean", rmse_mean)
    else:
        save_path = save_cfg.save_dir
        
    yaml = YAML(typ="safe")
    yaml.indent(mapping=2, sequence=4, offset=2)  # インデントの調整
    with open(os.path.join(save_path, "data.yaml"), "r") as f:
        data = yaml.load(f)

    data["metrics"]["rmse_mean"] = rmse_mean

    data = to_yaml_safe(data)

    with open(os.path.join(save_path, "data.yaml"), "w") as f:
        yaml.dump(data, f)

"""
### モデル作成時のデータを保存する ###
def save_create_data(
    model,
    scaler:StandardScaler,
    history_figure,
    training_time,
    save_cfg: SaveConfig,
    simulation_cfg: SimulationConfig,
    rnn_cfg: RnnConfig,
):
    fading_params = cfg_to_flat_dict(simulation_cfg)
    rnn_params = cfg_to_flat_dict(rnn_cfg)
    if save_cfg.use_mlflow:
        print("mlflowに保存します")
        import mlflow

        mlflow.set_experiment(save_cfg.experiment_name)

        with mlflow.start_run(run_name=save_cfg.run_name) as run:
            # simulation_cfg を全部保存
            for k, v in fading_params.items():
                mlflow.log_param(k, v)

            # rnn_cfg を全部保存
            for k, v in rnn_params.items():
                mlflow.log_param(k, v)

            mlflow.log_figure(history_figure, "loss_curve.png")
            mlflow.log_metric("training_time", training_time)
            artifact_dir = mlflow.get_artifact_uri()
            artifact_path = artifact_dir.replace("file:", "")
            
            joblib.dump(scaler, os.path.join(artifact_path, "scaler.pkl"))
            model_path = os.path.join(artifact_dir.replace("file:", ""), "model.keras")
            model.save(model_path)
            run_id = run.info.run_id
        print("実験名(experiment_name):", save_cfg.experiment_name)
        print("実験id:", run.info.experiment_id)
        print("実行名(run_name):", save_cfg.run_name)
        print("実行id:", run_id)
        return run_id
    else:
        print("jsonで保存します")
        os.makedirs(save_cfg.save_dir, exist_ok=True)

        data = {
            "run_name": save_cfg.run_name,
            "datetime": datetime.now().isoformat(),
            "params": {
                **fading_params,
                **rnn_params,
            },
            "metrics": {
                "training_time": training_time,
            },
        }
        save_dir = save_cfg.save_dir
        with open(os.path.join(save_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=2)
        history_figure.savefig(os.path.join(save_dir, "loss_curve.png"))
        joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
        model.save(os.path.join(save_dir, "model.keras"))
        print("experiment_name:", save_cfg.experiment_name)
        print("run_name:", save_cfg.run_name)
        return save_cfg.run_name


def save_predict_data(
    run_id,
    true_data,
    predict_data,
    rmse,
    predict_result_fig,
    rmse_mean,
    save_cfg: SaveConfig,
):

    if save_cfg.use_mlflow:
        print("mlflowに保存します")
        import mlflow

        with mlflow.start_run(run_id):
            artifact_dir = mlflow.get_artifact_uri()
            artifact_path = artifact_dir.replace("file:", "")

            true_path = os.path.join(artifact_path, "true.npy")
            pred_path = os.path.join(artifact_path, "predicted.npy")
            np.save(true_path, true_data)
            np.save(pred_path, predict_data)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("rmse_mean", rmse_mean)
            mlflow.log_figure(predict_result_fig, "predict_results.png")
        print("実験名(experiment_name):", save_cfg.experiment_name)
        print("実行id(run_id):", run_id)

    else:
        save_dir = save_cfg.save_dir
        print("jsonで保存します")
        with open(f"{save_dir}/data.json", "r") as f:
            data = json.load(f)
        data["metrics"]["rmse"] = rmse
        data["metrics"]["rmse_mean"] = rmse_mean

        with open(f"{save_dir}/data.json", "w") as f:
            json.dump(data, f, indent=2)
        predict_result_fig.savefig(f"{save_dir}/predict_results.png")
        np.save(f"{save_dir}/true.npy", true_data)
        np.save(f"{save_dir}/predicted.npy", predict_data)
        print("実験名(experiment_name):", save_cfg.experiment_name)
        print("実行名(run_id(name)):", run_id)
"""
