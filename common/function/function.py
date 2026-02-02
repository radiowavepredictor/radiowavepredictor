import numpy as np
from enum import Enum
from dataclasses import asdict, is_dataclass
from pydantic import BaseModel
from itertools import product
from copy import deepcopy

from keras.utils import timeseries_dataset_from_array
import tensorflow as tf

from common.schema import RnnConfig

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