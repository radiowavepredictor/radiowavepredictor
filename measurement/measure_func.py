import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array

from common.common_func import cfg_to_flat_dict
from measurement.configs.measure_schema import MeasureConfig, RnnConfig, SaveConfig

def read_csv(read_cource, measure_cfg: MeasureConfig):
    csv_path = f"./measurement/result/WAVE{read_cource:04d}/result_n{'t' if measure_cfg.data_axis=='time' else 'd'}-001.csv"
    csv_data = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
    csv_data = csv_data.iloc[
        int(len(csv_data) * measure_cfg.start_ratio) : int(
            len(csv_data) * measure_cfg.end_ratio
        )
    ]  # データから使う範囲を切り取る
    np_data = csv_data.values.astype(np.float64)
    return np_data


def multiple_csv_to_dataset(
    read_cources,
    input_len,
    measure_cfg: MeasureConfig,
    scaler: StandardScaler | None = None,
):
    # csv読み込み
    measure_data_arr = []
    for cource in read_cources:
        measure_data = read_csv(cource, measure_cfg)
        measure_data_arr.append(measure_data)

    # 標準化 
    if scaler is None:
        # データ配列を一つにつなげる(標準化の計算を行うため)
        data_flatten = np.concatenate(measure_data_arr)
        scaler = StandardScaler()
        data_norm_arr = scaler.fit(data_flatten)

    data_norm_arr = []
    for measure_data in measure_data_arr: # measure_dataはnp化できないので、for分で各行を正規化する
        data_norm_arr.append(scaler.transform(measure_data))

    # 配列の各行をデータセット化する
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

    # データセットをつなげて一つにする
    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)

    return dataset, scaler


def load_learning_dataset(measure_cfg: MeasureConfig, rnn_cfg: RnnConfig):
    train_dataset, scaler = multiple_csv_to_dataset(
        measure_cfg.train_cources,
        rnn_cfg.input_len,
        measure_cfg,
    )
    train_dataset = (
        train_dataset.shuffle(buffer_size=10000)
        .batch(rnn_cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset, scaler = multiple_csv_to_dataset(
        measure_cfg.val_cources, rnn_cfg.input_len, measure_cfg, scaler
    )
    val_dataset = val_dataset.batch(rnn_cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    return (train_dataset, val_dataset), scaler

def save_create_data(
    model,
    scaler,
    history_figure,
    training_time,
    measure_cfg: MeasureConfig,
    rnn_cfg: RnnConfig,
    save_cfg: SaveConfig,
):
    measure_params = cfg_to_flat_dict(measure_cfg)
    rnn_params = cfg_to_flat_dict(rnn_cfg)

    if save_cfg.use_mlflow:
        print("mlflowに保存します")
        import mlflow

        mlflow.set_experiment(save_cfg.experiment_name)
        with mlflow.start_run(run_name=save_cfg.run_name) as run:
            # measure_cfg を全部保存
            for k, v in measure_params.items():
                mlflow.log_param(k, v)

            # rnn_cfg を全部保存
            for k, v in rnn_params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("training_time", training_time)
            mlflow.log_figure(history_figure, "loss_curve.png")

            artifact_dir = mlflow.get_artifact_uri()
            artifact_path = artifact_dir.replace("file:", "")

            joblib.dump(scaler, os.path.join(artifact_path, "scaler.pkl"))
            model_path = os.path.join(artifact_path, "model.keras")
            model.save(model_path)

            run_id = run.info.run_id

        print("実験名(experiment_name):", save_cfg.experiment_name)
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
                **measure_params,
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
            mlflow.log_figure(predict_result_fig, "predict_results.png")
        print("実験名(experiment_name):", save_cfg.experiment_name)
        print("実行id(run_id):", run_id)

    else:
        save_dir = save_cfg.save_dir
        print("jsonで保存します")
        with open(f"{save_dir}/data.json", "r") as f:
            data = json.load(f)
        data["metrics"]["rmse"] = rmse

        with open(f"{save_dir}/data.json", "w") as f:
            json.dump(data, f, indent=2)
        predict_result_fig.savefig(f"{save_dir}/predict_results.png")
        np.save(f"{save_dir}/true.npy", true_data)
        np.save(f"{save_dir}/predicted.npy", predict_data)
        print("実験名(experiment_name):", save_cfg.experiment_name)
        print("実行名(run_id(name)):", run_id)
