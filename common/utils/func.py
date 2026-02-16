import numpy as np
from enum import Enum
from itertools import product
from copy import deepcopy

from keras.utils import timeseries_dataset_from_array
import tensorflow as tf

from common.schema.config import RnnConfig

def dbm_to_mw(dbm):
    return 10 ** (dbm / 10)

def mw_to_dbm(mw):
    return 10 * np.log10(mw)
    
def make_dataset(changed_data, input_len):
    '''
    データからRNN用のデータセットを生成する関数
    timeseries_dataset_from_array()と使い分ける
    make_data_setはnumpy配列を返すので、後から加工しやすい
    timeseries_dataset_from_arrayはtf.data.Datasetオブジェクトを返すので、prefetchなどtensorflow専用の関数が使える
    '''
    data, target = [], []

    for i in range(len(changed_data) - input_len):
        data.append(changed_data[i : i + input_len])
        target.append(changed_data[i + input_len])

    # RNN用に3次元のデータに変更する
    re_data = np.array(data).reshape(len(data), input_len, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return (re_data, re_target)
    
# ネストした辞書型を1次元にする
def flatten_dict(d: dict, sep: str = "/") -> dict:
    result = {}

    def walk(current: dict, prefix: str = ""):
        for k, v in current.items():
            new_key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, dict):
                walk(v, new_key)
            else:
                if new_key in result:
                    raise KeyError(f"Duplicate key detected: {new_key}")
                result[new_key] = v

    walk(d)
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
    
# 辞書型を最初の(1段目の?)階層で直積する
def build_section_grid(section: dict):
    fixed = {}
    grid_keys = []
    grid_values = []

    for key, value in section.items():
        if isinstance(value, list):
            grid_keys.append(key)
            grid_values.append(value)
        else:
            fixed[key] = value

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
    時系列のarray_of_arrayからkerasのmodel.fit()に渡せるようにデータセットを作る。
    各行が一つの時系列データであることを想定しているので、データ同士が干渉しないよう、
    各行ごとにdatasetにする。
    最後に全部つなげて1次元にして返却する。
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
        dataset.shuffle(buffer_size=100000)
        .batch(rnn_cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset

# 予測波形、予測元波形を同時にplotするときの、予測波形のplot設定を行う関数
def predict_plot_setting(input_len,sampling_rate,base_plot_start,base_plot_range,out_steps_num):
    # 予測波形を基準での順番と、予測元波形を基準での順番があるので注意
    # ↑のときは接頭辞をpr,     ↑のときは接頭辞無し にする
    predict_start = input_len+out_steps_num-1 # 予測波形は予測元波形のどこから予測を開始してるか
    
    plot_is_before_predict=base_plot_start<predict_start

    if plot_is_before_predict:
        predict_plot_range = base_plot_range - (predict_start-base_plot_start)
        pr_predict_index_start = 0
        predict_plot_start = predict_start 
    else:
        predict_plot_range =base_plot_range
        pr_predict_index_start = base_plot_start-predict_start
        predict_plot_start = base_plot_start
 
    # 予測データの何番目から何番目までのデータを使うか
    pr_predict_index=slice(pr_predict_index_start,pr_predict_index_start+predict_plot_range,1)
    # 予測データをplotするときのx軸の表示範囲の計算
    x_arange = np.arange(predict_plot_start, predict_plot_start+predict_plot_range) * sampling_rate
    
    return x_arange,pr_predict_index
