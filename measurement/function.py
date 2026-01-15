import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from common.schema import RnnConfig
from common.function.function import array_of_array_to_dataset
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

#??? scalerは渡さないといけないようにして、fitを関数外で行うようにしたほうがいいかも
def multiple_csv_to_dataset(
    read_cources,
    rnn_cfg:RnnConfig,
    measure_cfg: MeasureConfig,
    scaler: StandardScaler | None = None,
):
    """
    Parameters
    ----------
    scaler: StandardScaler | None
        scalerを渡さない場合、scalerを作って、そのとき関数内で使うデータでfit-transformする
        渡されている場合、transformのみを行う
    Returns
    ----------
    """
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
        scaler.fit(data_flatten)
        
    data_norm_arr = []
    for measure_data in measure_data_arr: # measure_dataはnp化できないので、for分で各行を正規化する
        data_norm_arr.append(scaler.transform(measure_data))

    dataset=array_of_array_to_dataset(data_norm_arr,rnn_cfg)

    return dataset, scaler


def load_learning_dataset(measure_cfg: MeasureConfig, rnn_cfg: RnnConfig):
    train_dataset,scaler=multiple_csv_to_dataset(measure_cfg.cource.train,rnn_cfg,measure_cfg)
    val_dataset,scaler=multiple_csv_to_dataset(measure_cfg.cource.val,rnn_cfg,measure_cfg,scaler)
    return (train_dataset, val_dataset), scaler