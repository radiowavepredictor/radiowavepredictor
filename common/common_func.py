import numpy as np
import matplotlib.pyplot as plt
import os
path = os.path.dirname(__file__)
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils import timeseries_dataset_from_array
import time

def normalize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)
    
def denormalize(normalized_data,base_data):
    return normalized_data * base_data.std(axis=0) + base_data.mean(axis=0)

def dbm_to_mw(dbm):
    return 10**(dbm / 10)

def mw_to_dbm(mw):
    return 10*np.log10(mw)

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
    data = (normalized_data - feature_range[0]) * (base_data_max - base_data_min) / scale + base_data_min
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
    verbose='auto', #ログの表示設定
    ):
    
    print("🚀 新しいモデルを作成します")
    model = Sequential()
    model.add(Input(shape=(input_len, in_features)))

    for hidden_num in hidden_nums[:-1]:
        model.add(rnn_class(hidden_num, return_sequences=True))
        #model.add(USE_RNN_LAYER(hidden_num, return_sequences=True,kernel_regularizer=l2(1e-5))) L2正則化をするときはこっちを使う
    model.add(rnn_class(hidden_nums[-1], return_sequences=False))
    #model.add(USE_RNN_LAYER(HIDDEN_NUMS[-1], return_sequences=False,kernel_regularizer=l2(1e-5)))
    model.add(Dense(out_steps_num))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer=optimizer_class(learning_rate=learning_rate))
    model.summary()

    start_time=time.time()
    history=model.fit(
        dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience=10)],
        verbose=verbose, 
    )

    end_time=time.time()
    training_time=end_time-start_time
    
    history_figure=plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()

    return {
        "history_figure": history_figure,
        "training_time": training_time,
        "model": model,
    }

def predict(
    model,
    data,
    input_len,
    plot_start,
    plot_range
    ):
    
    power = np.abs(data) ** 2
    power_db = 10 * np.log10(power)
    normalized_data = normalize(power_db)

    x=timeseries_dataset_from_array(
        normalized_data,
        targets=None,
        sequence_length=input_len,
        batch_size=1,
        shuffle=False
    )

    predicted = model.predict(x)
    true_data=power_db
    denormalized_predicted = denormalize(predicted,true_data)
    reshape_denormalized_predicted = np.array(denormalized_predicted).reshape(len(denormalized_predicted)) #RMSEを出すために、true_dataと同じ形式にする
    rmse=np.sqrt(np.mean((reshape_denormalized_predicted[:-1]-true_data[input_len:])**2))

    # plotするときに単位を秒にするための処理
    # ???ここから分かりづらいかも 変数名も処理も分かりやすくしたい
    x_true_data=np.linspace(plot_start/20,(plot_start+plot_range)/20,plot_range)
    x_predict=np.linspace((plot_start+input_len)/20,(plot_start+plot_range)/20,plot_range-input_len)

    predict_result_fig=plt.figure()
    plt.xlabel("Time[s]")
    plt.ylabel("ReceivedPower[dBm]")
    plt.plot(x_true_data,true_data[plot_start:plot_start+plot_range],color="r",alpha=0.5,label="true_data")
    plt.plot(x_predict, denormalized_predicted[plot_start:plot_start+plot_range-input_len], color="g", label="predict_data")
    plt.legend()

    return {
        "rmse":rmse,
        "predict_result_figure":predict_result_fig,
        "true_data":true_data,
        "predict_data":denormalized_predicted        # ???reshape_denormalized_predictedのほうがいいのかもしれない
    }
 