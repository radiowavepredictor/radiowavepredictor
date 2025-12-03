import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import timeseries_dataset_from_array

from setting import *
from func import *

'''
csv_path= f"{path}/result/WAVE{PREDICT_COURCE:04d}/result_n{LEARN_MODE}-001.csv" 
data_csv = pd.read_csv(csv_path, usecols=IN_FEATURES)
'''
fading_data=calc_fading()
#fading_data=calc_nakagami_rice_fading()
power = np.abs(fading_data) ** 2
#power_db = 10 * np.log10(power)
#normalized_data = normalize(power_db)
#normalized_data = power_db
normalized_data = power
''' 
normalized_data = normalize(faging_data)
normalized_data = normalized_data.values.astype(np.float64)
'''
x=timeseries_dataset_from_array(
    normalized_data,
    targets=None,
    sequence_length=INPUT_LEN,
    batch_size=1,
    shuffle=None
)
#(x,y)=make_data_set(normalized_data,INPUT_LEN)

if os.path.exists(MODEL_PATH):
    print("✅ 既存のモデルを読み込みます")
    model = load_model(MODEL_PATH)
else:
    print("モデルが見つかりません")

predicted = model.predict(x)
#true_data = np.array(data_csv[OUT_FEATURES]).reshape(len(data_csv[OUT_FEATURES]))
#true_data=power_db
true_data=power
#denormalized_predicted = denormalize(predicted,true_data)
denormalized_predicted = predicted
denormalized_predicted = 10 * np.log10(denormalized_predicted)
reshape_denormalied_predeicted = np.array(denormalized_predicted).reshape(len(denormalized_predicted))
true_data = 10 * np.log10(true_data)
rmse=np.sqrt(np.mean((reshape_denormalied_predeicted[:-1]-true_data[INPUT_LEN:])**2))
print(rmse)
'''
##### ここから再帰予測 #####
input_data=normalized_data[-PREDICT_LEN-INPUT_LEN:-PREDICT_LEN]
future_result = np.empty((0,)) # (0,)で空の配列になる

for i in range(PREDICT_LEN):
    print(f'step:{i+1}')
    test_data=np.reshape(input_data,(1,INPUT_LEN,1))
    predicted=model.predict(test_data)

    input_data=np.delete(input_data,0)
    input_data=np.append(input_data,predicted)

    future_result=np.append(future_result,predicted)
#true_data = np.array(data_csv[OUT_FEATURES]).reshape(len(data_csv[OUT_FEATURES]))
#true_data=power_db
true_data=power
#denormalized_predicted = denormalize(predicted,true_data)
denormalized_predicted = future_result
denormalized_predicted = 10 * np.log10(denormalized_predicted)
reshape_denormalied_predeicted = np.array(denormalized_predicted).reshape(len(denormalized_predicted))
true_data = 10 * np.log10(true_data)
mse=np.mean((future_result - true_data[-PREDICT_LEN:])**2)
rmse=np.sqrt(np.mean((future_result - true_data[-PREDICT_LEN:])**2))
print(rmse)
'''


# ここで使うデータは0.05ミリ秒毎にサンプリングされている
# plotするときに単位を秒にするための準備
x_true_data=np.linspace(PLOT_START/20,(PLOT_START+PLOT_RANGE)/20,PLOT_RANGE)
x_predict=np.linspace((PLOT_START+INPUT_LEN)/20,(PLOT_START+PLOT_RANGE)/20,PLOT_RANGE-INPUT_LEN)
'''
if os.path.exists(f"{path}/predict_result.pkl"):
    with open(f'{path}/predict_result.pkl', 'rb') as f:
        predict_result = pickle.load(f)
else:
    print("predict_result.pklがないです")
    predict_result={
        INPUT_LEN:denormalized_predicted
    }
predict_result[INPUT_LEN]=denormalized_predicted
with open(f'{path}/predict_result.pkl', 'wb') as f:
    pickle.dump(predict_result, f)
'''
plt.figure()
plt.xlabel("Time[s]")
plt.ylabel("ReceivedPower[dBm]")
plt.plot(x_true_data,true_data[PLOT_START:PLOT_START+PLOT_RANGE],color="r",alpha=0.5,label="true_data")
plt.plot(x_predict, denormalized_predicted[PLOT_START:PLOT_START+PLOT_RANGE-INPUT_LEN], color="g", label="predict_data")
#plt.plot(range(len(true_data)-PREDICT_LEN, len(true_data)), future_result, color="g", label="future_predict") #再帰用
#plt.plot(range(len(true_data)-PLOT_RANGE-PREDICT_LEN,len(true_data)),true_data[-PLOT_RANGE-PREDICT_LEN:],color="r",alpha=0.5,label="true_data") #再帰用
plt.legend()
plt.show()
