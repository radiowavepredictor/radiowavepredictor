import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array

from common.schema import RnnConfig
from measurement.configs.schema import MeasureConfig

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