import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import asdict, is_dataclass

path = os.path.dirname(__file__)
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils import timeseries_dataset_from_array
import time
from sklearn.preprocessing import StandardScaler
'''
def z_score_normalize(data,base_data:np.ndarray):
    return (data - base_data.mean()) / base_data.std()

def z_score_denormalize(normalized_data, base_data:np.ndarray):
    return normalized_data * base_data.std() + base_data.mean()
'''    
def dbm_to_mw(dbm):
    return 10 ** (dbm / 10)

def mw_to_dbm(mw):
    return 10 * np.log10(mw)


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


def create_model(
    dataset,
    val_dataset,
    input_len,
    in_features,
    hidden_nums,
    rnn_class,
    optimizer_class,
    out_steps_num,
    learning_rate,
    epochs,
    verbose=1,  # ログの表示設定
):

    model = Sequential()
    model.add(Input(shape=(input_len, in_features)))

    for hidden_num in hidden_nums[:-1]:
        model.add(rnn_class(hidden_num, return_sequences=True))
        # model.add(USE_RNN_LAYER(hidden_num, return_sequences=True,kernel_regularizer=l2(1e-5))) L2正則化をするときはこっちを使う
    model.add(rnn_class(hidden_nums[-1], return_sequences=False))
    # model.add(USE_RNN_LAYER(HIDDEN_NUMS[-1], return_sequences=False,kernel_regularizer=l2(1e-5)))
    model.add(Dense(out_steps_num))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer=optimizer_class(learning_rate=learning_rate))
    if verbose == 1:
        model.summary()

    start_time = time.time()
    history = model.fit(
        dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[EarlyStopping(monitor="val_loss", mode="auto", patience=20)],
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
    
# config構造体を辞書型に変換して返す
def cfg_to_flat_dict(cfg):
    if not is_dataclass(cfg):
        raise TypeError("cfg must be a dataclass")

    result = {}
    for k, v in asdict(cfg).items():  # type:ignore[arg-type]
        if isinstance(v, type):
            result[k] = v.__name__
        elif hasattr(v, "__name__"):
            result[k] = v.__name__
        else:
            result[k] = v
    return result

