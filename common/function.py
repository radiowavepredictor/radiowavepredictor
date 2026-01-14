import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum
from dataclasses import asdict, is_dataclass
from pydantic import BaseModel
from ruamel.yaml import YAML
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
from itertools import product
from copy import deepcopy


from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils import timeseries_dataset_from_array
import time
from sklearn.preprocessing import StandardScaler

from common.schema import RnnConfig,SaveConfig

def dbm_to_mw(dbm):
    return 10 ** (dbm / 10)

def mw_to_dbm(mw):
    return 10 * np.log10(mw)

'''多分消していい
def min_max_normalize(data, feature_range=(0, 1)):
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    scale = feature_range[1] - feature_range[0]
    normalized = feature_range[0] + (data - data_min) * scale / (data_max - data_min)
    return normalized


def min_max_denormalize(normalized_data, base_data, feature_range=(0, 1)):
    base_data_min = base_data.min(axis=0)
    base_data_max = base_data.max(axis=0)
    scale = feature_range[1] - feature_range[0]
    data = (normalized_data - feature_range[0]) * (
        base_data_max - base_data_min
    ) / scale + base_data_min
    return data
'''

def create_model(
    dataset,
    val_dataset,
    rnn_cfg:RnnConfig,
    verbose=1,  # ログの表示設定
):

    model = Sequential()
    model.add(Input(shape=(rnn_cfg.input_len, rnn_cfg.in_features)))

    for hidden_num in rnn_cfg.hidden_nums[:-1]:
        model.add(rnn_cfg.rnn_class( hidden_num,return_sequences=True))
        # model.add(USE_RNN_LAYER(hidden_num, return_sequences=True,kernel_regularizer=l2(1e-5))) L2正則化をするときはこっちを使う(間違ってるかも)
    model.add(rnn_cfg.rnn_class(rnn_cfg.hidden_nums[-1], return_sequences=False))
    # model.add(USE_RNN_LAYER(HIDDEN_NUMS[-1], return_sequences=False,kernel_regularizer=l2(1e-5)))
    model.add(Dense(rnn_cfg.out_steps_num))
    model.add(Activation("linear"))
    optimizer=rnn_cfg.optimizer_class(learning_rate=rnn_cfg.learning_rate)
    model.compile(loss="mse", optimizer=optimizer) #type:ignore[arg-type]
    if verbose == 1:
        model.summary()

    start_time = time.time()
    history = model.fit(
        dataset,
        epochs=rnn_cfg.epochs,
        validation_data=val_dataset,
        callbacks=[EarlyStopping(monitor="val_loss", mode="auto", patience=rnn_cfg.patience)],
        verbose=verbose,  # type:ignore[arg-type]
    )

    end_time = time.time()
    training_time = end_time - start_time

    history_figure = plt.figure()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()

    return {
        "history_figure": history_figure,
        "training_time": training_time,
        "model": model,
    }


def predict(model, data:np.ndarray,scaler:StandardScaler, input_len, plot_start, plot_range,verbose=1):
    """
    Parameters
    ----------
    data : np.ndarray
        (データ,入力特徴量の数)の2次元行列を期待しています
        特徴量が一つの場合はreshape(-1,1)してください
    
    Returns
    ----------
    """
    norm_data = scaler.transform(data)

    x = timeseries_dataset_from_array(
        norm_data,
        targets=None,
        sequence_length=input_len,
        batch_size=32,
        shuffle=False,
    )

    predicted = model.predict(x, verbose=verbose)
    denormalized_predicted = scaler.inverse_transform(predicted)

    print(denormalized_predicted.shape)
    print(data.shape)
    rmse = np.sqrt(
        np.mean((denormalized_predicted[:-1] - data[input_len:]) ** 2)
    )

    # plotするときに単位を秒にするための処理
    # ???ここから分かりづらいかも 変数名も処理も分かりやすくしたい
    x_true_data = np.linspace(
        plot_start / 20, (plot_start + plot_range) / 20, plot_range
    )
    x_predict = np.linspace(
        (plot_start + input_len) / 20,
        (plot_start + plot_range) / 20,
        plot_range - input_len,
    )

    predict_result_fig = plt.figure()
    plt.xlabel("Time[s]")
    plt.ylabel("ReceivedPower[dBm]")
    plt.plot(
        x_true_data,
        data[plot_start : plot_start + plot_range],
        color="r",
        alpha=0.5,
        label="true_data",
    )
    plt.plot(
        x_predict,
        denormalized_predicted[plot_start : plot_start + plot_range - input_len],
        color="g",
        label="predict_data",
    )
    plt.legend()

    return {
        "rmse": rmse,
        "predict_result_figure": predict_result_fig,
        "true_data": data,
        "predict_data": denormalized_predicted,  # ???reshape_denormalized_predictedのほうがいいのかもしれない
    }
    
# dataからRNN用のデータセットを生成する関数
# timeseries_dataset_from_array()と使い分ける
# make_data_setはnumpy配列を返すので、後から加工しやすい
# timeseries_dataset_from_arrayはtf.data.Datasetオブジェクトを返すので、prefetchなどtensorflow専用の関数が使える
def make_dataset(changed_data, input_len):
    data, target = [], []

    for i in range(len(changed_data) - input_len):
        data.append(changed_data[i : i + input_len])
        target.append(changed_data[i + input_len])

    # RNN用に3次元のデータに変更する
    re_data = np.array(data).reshape(len(data), input_len, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return (re_data, re_target)
    
# 構造体を辞書型に変換してflatにして返す Enumがあった場合は文字列に変換する
def struct_to_flat_dict(obj)->dict:
    if isinstance(obj,dict):
        return obj
    if is_dataclass(obj): #???多分他の構造体もあるから汎用的なものを探すかどれかに絞ったほうがいい
        dict_=asdict(obj) #type:ignore[arg--type]
    elif isinstance(obj,BaseModel):
        dict_=obj.model_dump()
    else:
        raise TypeError("dataclassまたはBaseModel以外の値が入っています")

    result={}
    def walk_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk_(v)
            else:
                result[k] = v

    walk_(dict_)
    return result

def to_yaml_safe(value:dict)->dict:
    """
    入れ子の構造を再帰的に走査し、YAMLに渡せる型に変換する
    構造体がネストしていると、__name__に変わってしまうため、基本的にはflattenしてから使う
    - Enum -> .value
    - クラス / 構造体 -> .__name__
    - NumPy配列 -> list
    - NumPyスカラー -> Python型に変換
    - set / frozenset / tuple -> list
    - bytes -> utf-8文字列
    - complex -> [real, imag]
    """
    if isinstance(value, Enum):
        return value.value
    elif hasattr(value, "__name__") and not isinstance(value, type):
        # クラスや構造体のインスタンス
        return value.__name__ #type:ignore[arg-type]
    elif isinstance(value, dict):
        return {k: to_yaml_safe(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple, set, frozenset)):
        return [to_yaml_safe(v) for v in value]
    elif isinstance(value, np.ndarray):
        return [to_yaml_safe(v) for v in value.tolist()]
    elif isinstance(value, np.generic):
        # NumPy のスカラー型を Python 標準型に変換
        return value.item()
    elif isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return str(value)
    elif isinstance(value, complex):
        return [value.real, value.imag]
    else:
        return value

def save_create_data(
    model,
    scaler,
    history_figure,
    training_time,
    data_cfg,
    rnn_cfg: RnnConfig,
    save_cfg: SaveConfig,
):
    """
    Parameters
    ----------
    data_cfg 
        学習時に使ったデータの設定構造体、もしくは辞書型を期待しています。
    
    Returns
    ----------
    """
    data_params = struct_to_flat_dict(data_cfg)
    data_params = to_yaml_safe(data_params)
    rnn_params = struct_to_flat_dict(rnn_cfg)
    rnn_params = to_yaml_safe(rnn_params)

    if save_cfg.use_mlflow:
        import mlflow

        mlflow.set_experiment(save_cfg.experiment_name)
        with mlflow.start_run(run_name=save_cfg.run_name) as run:
            # measure_cfg を全部保存
            for k, v in data_params.items():
                mlflow.log_param(k, v)

            # rnn_cfg を全部保存
            for k, v in rnn_params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("training_time", training_time)
            mlflow.log_figure(history_figure, "loss_curve.png")

            artifact_dir = mlflow.get_artifact_uri()
            save_path = artifact_dir.replace("file:", "")

            run_id = run.info.run_id
    else:
        os.makedirs(save_cfg.save_dir, exist_ok=True)

        save_path = save_cfg.save_dir
        history_figure.savefig(os.path.join(save_path, "loss_curve.png"))

        run_id = save_cfg.run_name
        
    data = {
        "run_name": save_cfg.run_name,
        "datetime": datetime.now().isoformat(),
        "params": {
            **data_params,
            **rnn_params,
        },
        "metrics": {
            "training_time": training_time,
        },
    }
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)  # インデントの調整
    with open(os.path.join(save_path, "data.yaml"), "w") as f:
        yaml.dump(data, f)

    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
    model.save(os.path.join(save_path, "model.keras"))
    
    return run_id
    
def save_predict_data(
    run_id,
    true_data,
    predict_data,
    rmse,
    predict_result_fig,
    save_cfg: SaveConfig,
):

    if save_cfg.use_mlflow:
        import mlflow

        with mlflow.start_run(run_id):
            artifact_dir = mlflow.get_artifact_uri()
            save_path = artifact_dir.replace("file:", "")

            mlflow.log_metric("rmse", rmse)
            mlflow.log_figure(predict_result_fig, "predict_results.png")

    else:
        save_path = save_cfg.save_dir
        predict_result_fig.savefig(os.path.join(save_path),"predict_results.png")
        
    yaml = YAML(typ="safe")
    yaml.indent(mapping=2, sequence=4, offset=2)  # インデントの調整
    with open(os.path.join(save_path, "data.yaml"), "r") as f:
        data=yaml.load(f)
        
    data["metrics"]["rmse"] = rmse
    data=to_yaml_safe(data)
    
    with open(os.path.join(save_path, "data.yaml"), "w") as f:
        yaml.dump(data, f)
    
    np.save(os.path.join(save_path, "true.npy"), true_data)
    np.save(os.path.join(save_path, "predicted.npy"), predict_data)
    
# 辞書型からを最初の(1段目の?)階層で直積する
def build_section_grid(section: dict):
    fixed = {}
    grid_keys = []
    grid_values = []

    for k, v in section.items():
        if isinstance(v, list):
            grid_keys.append(k)
            grid_values.append(v)
        else:
            fixed[k] = v

    # グリッドがない場合も product が回るように
    if not grid_keys:
        return [fixed]

    results = []
    for values in product(*grid_values):
        d = deepcopy(fixed)
        for k, v in zip(grid_keys, values):
            d[k] = v
        results.append(d)

    return results

