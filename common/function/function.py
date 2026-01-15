import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import asdict, is_dataclass
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from itertools import product
from copy import deepcopy
import time

from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils import timeseries_dataset_from_array
import tensorflow as tf

from common.schema import RnnConfig

def dbm_to_mw(dbm):
    return 10 ** (dbm / 10)

def mw_to_dbm(mw):
    return 10 * np.log10(mw)

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


def predict(model, data:np.ndarray,scaler:StandardScaler, rnn_cfg:RnnConfig, plot_start, plot_range,verbose=1):
    """
    Parameters
    ----------
    data : np.ndarray
        (データ,入力特徴量の数)の2次元行列を期待しています
        特徴量が一つの場合はreshape(-1,1)してください
    
    Returns
    rmse_arr : list[float]
        out_steps_num分のrmseの配列
    ----------
    """
    norm_data = scaler.transform(data)

    x = timeseries_dataset_from_array(
        norm_data,
        targets=None,
        sequence_length=rnn_cfg.input_len,
        batch_size=32,
        shuffle=False,
    )

    predicted = model.predict(x, verbose=verbose)
    denormalized_predicted = scaler.inverse_transform(predicted)

    print(denormalized_predicted.shape)
    print(data.shape)
    rmse_arr = np.array([
        np.sqrt(
            np.mean((denormalized_predicted[:-i-1,i].reshape(-1,1) - data[rnn_cfg.input_len+i:]) ** 2)
        )
        for i in range(rnn_cfg.out_steps_num)
    ])
    
    # plotするときに単位を秒にするための処理
    # ???ここから分かりづらいかも 変数名も処理も分かりやすくしたい
    x_true_data = np.linspace(
        plot_start / 20, (plot_start + plot_range) / 20, plot_range
    )
    x_predict = np.linspace(
        (plot_start + rnn_cfg.input_len) / 20,
        (plot_start + plot_range) / 20,
        plot_range - rnn_cfg.input_len,
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
        denormalized_predicted[plot_start : plot_start + plot_range - rnn_cfg.input_len],
        color="g",
        label="predict_data",
    )
    plt.legend()

    return {
        "rmse_arr": rmse_arr,
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
    中で構造体がネストしていると、__name__に変わってしまうため、基本的にはflattenしてから使う
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

def make_target(data,input_len,out_steps_num):
    """
    N点予測に合わせて教師データを作成する
    """
    targets = np.stack(
        [
            data[i : i + out_steps_num]
            for i in range(input_len, len(data) - out_steps_num + 1)
        ],
        axis=0,
    )
    return targets

def array_of_array_to_dataset(arr_of_arr,rnn_cfg:RnnConfig):
    """
    時系列のarray_of_arrayからkerasのmodel.fit()に渡せるようにデータセットを作る
    各行が一つの時系列データであることを想定しているので、データ同士が干渉しないよう、
    各行ごとにdatasetにして、最後にそれをつなげて1次元にしている
    """
    # 配列の各行をデータセット化する
    dataset_arr = []
    for inner_arr in arr_of_arr:
        targets=make_target(inner_arr,rnn_cfg.input_len,rnn_cfg.out_steps_num)
        # datasetの中身はtensorflow特有のオブジェクトで入力と出力(入力に対する答え)のセットが入っている
        dataset_i = timeseries_dataset_from_array(
            data=inner_arr,
            targets=targets,
            sequence_length=rnn_cfg.input_len,
            batch_size=None,  # type:ignore[arg-type]
            shuffle=False,
        )
        dataset_arr.append(dataset_i)

    # データセットをつなげて一つにする
    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)
        
    dataset = (
        dataset.shuffle(buffer_size=10000)
        .batch(rnn_cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset
