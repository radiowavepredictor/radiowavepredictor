import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
from dataclasses import dataclass

from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils import timeseries_dataset_from_array

from common.schema.config import RnnConfig
from common.function.func import predict_plot_setting


def create_model(
    dataset,
    val_dataset,
    rnn_cfg: RnnConfig,
    verbose=1,  # ログの表示設定
):

    model = Sequential()
    model.add(Input(shape=(rnn_cfg.input_len, rnn_cfg.in_features)))

    for hidden_num in rnn_cfg.hidden_nums[:-1]:
        model.add(rnn_cfg.rnn_class(hidden_num, return_sequences=True))
        # model.add(USE_RNN_LAYER(hidden_num, return_sequences=True,kernel_regularizer=l2(1e-5))) L2正則化をするときはこっちを使う(間違ってるかも)
    model.add(rnn_cfg.rnn_class(rnn_cfg.hidden_nums[-1], return_sequences=False))
    # model.add(USE_RNN_LAYER(HIDDEN_NUMS[-1], return_sequences=False,kernel_regularizer=l2(1e-5)))
    model.add(Dense(rnn_cfg.out_steps_num))
    model.add(Activation("linear"))
    optimizer = rnn_cfg.optimizer_class(learning_rate=rnn_cfg.learning_rate)
    model.compile(loss="mse", optimizer=optimizer)  # type:ignore[arg-type]
    if verbose == 1:
        model.summary()

    start_time = time.time()
    history = model.fit(
        dataset,
        epochs=rnn_cfg.epochs,
        validation_data=val_dataset,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="auto", patience=rnn_cfg.patience)
        ],
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


@dataclass
class PredictResult():
    true_data: np.ndarray
    predict_data: dict
    predict_figure: dict
    rmse: dict
    predict_time: float

def predict(
    model,
    data: np.ndarray,
    scaler: StandardScaler,
    rnn_cfg: RnnConfig,
    plot_start,
    plot_range,
    sampling_rate,
    verbose=1,
)->PredictResult:
    """
    Parameters
    ----------
    data : np.ndarray
        (各データ長,入力特徴量の数)の2次元行列を期待しています
        特徴量が一つの場合はreshape(-1,1)してください
    ----------
    """
    norm_data = scaler.transform(data)

    x = timeseries_dataset_from_array(
        norm_data,
        targets=None,
        sequence_length=rnn_cfg.input_len,
        batch_size=1,
        shuffle=False,
    )

    start_time = time.time()
    predicted = model.predict(x, verbose=verbose)
    end_time = time.time()
    predict_time = end_time - start_time
    denormalized_predicted = scaler.inverse_transform(predicted)

    predicted_arr = [
        denormalized_predicted[: -i - 1, i].reshape(-1, 1)
        for i in range(rnn_cfg.out_steps_num)
    ]
    predict_data_dict = {f"step-{i+1}": value for i, value in enumerate(predicted_arr)}
    # 予測ステップ数分のrmseをつくる
    rmse_arr = np.array(
        [
            np.sqrt(np.mean((predicted_arr[i] - data[rnn_cfg.input_len + i :]) ** 2))
            for i in range(rnn_cfg.out_steps_num)
        ]
    )
    rmse_dict = {f"rmse-{i+1}": v for i, v in enumerate(rmse_arr)}

    x_arange_true = np.arange(plot_start, plot_start + plot_range) * sampling_rate
    # plotするときに単位を距離にするための処理
    predict_fig_dict = {}

    for i in range(rnn_cfg.out_steps_num):
        x_arange_predict, predict_index = predict_plot_setting(
            rnn_cfg.input_len, sampling_rate, plot_start, plot_range, i + 1
        )

        fig = plt.figure()

        plt.xlabel("Time[s]")
        plt.ylabel("ReceivedPower[dBm]")

        # true_data
        plt.plot(
            x_arange_true,
            data[plot_start : plot_start + plot_range],
            color="r",
            alpha=0.5,
            label="true_data",
        )
        # predict_data (iステップ目)
        plt.plot(
            x_arange_predict,
            denormalized_predicted[predict_index, i],
            color="g",
            label=f"predict_step_{i+1}",
        )

        plt.legend()
        plt.title(f"Prediction Step {i+1}")

        predict_fig_dict[f"step-{i+1}"] = fig

    return PredictResult(
        rmse=rmse_dict,
        true_data= data,
        predict_data=predict_data_dict,
        predict_figure= predict_fig_dict,
        predict_time= predict_time,
    )
