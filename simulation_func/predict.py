import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import timeseries_dataset_from_array

from simulation_func.setting import *
from simulation_func.simu_func import *
from func import *

fading_data=calc_nakagami_rice_fading()
power = np.abs(fading_data) ** 2
power_db = 10 * np.log10(power)
normalized_data = normalize(power_db)

x=timeseries_dataset_from_array(
    normalized_data,
    targets=None,
    sequence_length=INPUT_LEN,
    batch_size=1,
    shuffle=None
)

if os.path.exists(MODEL_PATH):
    print("✅ 既存のモデルを読み込みます")
    model = load_model(MODEL_PATH)
else:
    print("モデルが見つかりません")

predicted = model.predict(x)
true_data=power_db
denormalized_predicted = denormalize(predicted,true_data)
reshape_denormalied_predeicted = np.array(denormalized_predicted).reshape(len(denormalized_predicted))
rmse=np.sqrt(np.mean((reshape_denormalied_predeicted[:-1]-true_data[INPUT_LEN:])**2))
print(rmse)


# ここで使うデータは0.05ミリ秒毎にサンプリングされている
# plotするときに単位を秒にするための準備
x_true_data=np.linspace(PLOT_START/20,(PLOT_START+PLOT_RANGE)/20,PLOT_RANGE)
x_predict=np.linspace((PLOT_START+INPUT_LEN)/20,(PLOT_START+PLOT_RANGE)/20,PLOT_RANGE-INPUT_LEN)

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
    
plt.figure()
plt.xlabel("Time[s]")
plt.ylabel("ReceivedPower[dBm]")
plt.plot(x_true_data,true_data[PLOT_START:PLOT_START+PLOT_RANGE],color="r",alpha=0.5,label="true_data")
plt.plot(x_predict, denormalized_predicted[PLOT_START:PLOT_START+PLOT_RANGE-INPUT_LEN], color="g", label="predict_data")
plt.legend()
plt.show()
