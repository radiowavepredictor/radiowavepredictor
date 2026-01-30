import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils import timeseries_dataset_from_array

from common.schema import RnnConfig


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
        batch_size=1,
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
 