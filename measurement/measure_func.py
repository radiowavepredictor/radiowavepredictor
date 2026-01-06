import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array
from dataclasses import asdict, is_dataclass
import json
from datetime import datetime
from typing import Any

from common.common_func import normalize
from measurement.configs.meas_schema import MeasureConfig,RnnConfig,SaveConfig

def compute_train_mean_std(read_cources, measure_cfg):
    all_data = []

    for cource in read_cources:
        csv_path = f"./measurement/result/WAVE{cource:04d}/result_n{'t' if measure_cfg.data_axis=='time' else 'd'}-001.csv"
        csv_data = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
        csv_data = csv_data.iloc[
            int(len(csv_data)*measure_cfg.start_ratio):
            int(len(csv_data)*measure_cfg.end_ratio)
        ]
        all_data.append(csv_data.values.astype(np.float64))

    all_data = np.concatenate(all_data, axis=0)
    return all_data.mean(), all_data.std()
    
def csv_to_dataset(csv_path, input_len, start_ratio, end_ratio, mean, std):
    csv_data = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
    csv_data = csv_data.iloc[
        int(len(csv_data)*start_ratio):
        int(len(csv_data)*end_ratio)
    ]
    np_data = csv_data.values.astype(np.float64)

    data_normalized = (np_data - mean) / std

    dataset = timeseries_dataset_from_array(
        data=data_normalized,
        targets=data_normalized[input_len:],
        sequence_length=input_len,
        batch_size=None,
        shuffle=None
    )
    return dataset
    
def multiple_csv_to_dataset(read_cources, input_len, measure_cfg, mean, std):
    dataset_arr = []

    for cource in read_cources:
        csv_path = f"./measurement/result/WAVE{cource:04d}/result_n{'t' if measure_cfg.data_axis=='time' else 'd'}-001.csv"
        ds = csv_to_dataset(
            csv_path,
            input_len,
            measure_cfg.start_ratio,
            measure_cfg.end_ratio,
            mean,
            std
        )
        dataset_arr.append(ds)

    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)

    return dataset
    
def load_training_data(measure_cfg: MeasureConfig, rnn_cfg: RnnConfig):
    mean, std = compute_train_mean_std(
        measure_cfg.train_corces,
        measure_cfg
    )

    train_dataset = multiple_csv_to_dataset(
        measure_cfg.train_corces,
        rnn_cfg.input_len,
        measure_cfg,
        mean,
        std
    )
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=10000)
        .batch(rnn_cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = multiple_csv_to_dataset(
        measure_cfg.val_corces,
        rnn_cfg.input_len,
        measure_cfg,
        mean,
        std
    )
    val_dataset = val_dataset.batch(rnn_cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset,mean,std

'''
def csv_to_dataset(csv_path,input_len,start_ratio,end_ratio): 
    csv_data = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
    # start_ratioからend_ratioまででデータを切り取る
    csv_data = csv_data.iloc[int(len(csv_data)*start_ratio):int(len(csv_data) * end_ratio)] 
    np_data = csv_data.values.astype(np.float64)

    data_normalized = normalize(np_data)

    #datasetの中身はtensorflow特有のオブジェクトで入力と出力(入力に対する答え)のセットが入っている    
    dataset=timeseries_dataset_from_array(
        data=data_normalized,
        targets=data_normalized[input_len:],
        sequence_length=input_len,
        batch_size=None,
        shuffle=None
    )
    return dataset 

def multiple_csv_to_dataset(read_cources,input_len,measure_cfg:MeasureConfig):
    dataset_arr=[]
    for cource in read_cources:
        csv_path = f"./measurement/result/WAVE{cource:04d}/result_n{"t" if measure_cfg.data_axis=="time" else "d" }-001.csv" #???ここなんかやばいから絶対変える
        train_dataset_i=csv_to_dataset(csv_path,input_len,measure_cfg.start_ratio,measure_cfg.end_ratio)

        dataset_arr.append(train_dataset_i)

    # dataset_arrの中身をすべてつなげる
    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)
    return dataset

#csvからデータを読み込んで機械学習の訓練データセットと検証データセットを返す関数
def load_training_data(measure_cfg:MeasureConfig,rnn_cfg:RnnConfig):
    train_dataset=multiple_csv_to_dataset(measure_cfg.train_corces,rnn_cfg.input_len,measure_cfg)
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=10000)
        .batch(rnn_cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    val_dataset =multiple_csv_to_dataset(measure_cfg.val_corces,rnn_cfg.input_len,measure_cfg)
    val_dataset = val_dataset.batch(rnn_cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset,val_dataset
'''
#dataからRNN用のデータセットを生成する関数
#timeseries_dataset_from_array()と使い分ける
#make_data_setはnumpy配列を返すので、後から加工しやすい
#timeseries_dataset_from_arrayはtf.data.Datasetオブジェクトを返すので、prefetchなどtensorflow専用の関数が使える
def make_data_set(changed_data,input_len):
    data,target=[],[]
    
    for i in range(len(changed_data)-input_len):
        data.append(changed_data[i:i + input_len])
        target.append(changed_data[i + input_len])

    # RNN用に3次元のデータに変更する
    re_data = np.array(data).reshape(len(data), input_len, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return (re_data, re_target)


def cfg_to_flat_dict(cfg): 
    if not is_dataclass(cfg):
        raise TypeError("cfg must be a dataclass")

    result = {}
    for k, v in asdict(cfg).items():
        if isinstance(v, type):
            result[k] = v.__name__
        elif hasattr(v, "__name__"):
            result[k] = v.__name__
        else:
            result[k] = v
    return result

def save_create_data(
    model,
    history_figure,
    training_time,
    measure_cfg:MeasureConfig,
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
            model_path = os.path.join(
                artifact_dir.replace("file:", ""), "model.keras"
            )
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
