import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model

from common.function.function import predict
from common.function.save import save_predict_data
from measurement.configs.config import RNN_CFG,SAVE_CFG,MEASURE_CFG

# run_idの取得
with open("./measurement/scripts/run_id.txt", "r") as f:
    run_id = f.readline().strip()

# run_idで作ったmodelを探す
if SAVE_CFG.use_mlflow:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_path = client.download_artifacts(run_id, "model.keras")
    scaler_path = client.download_artifacts(run_id,"scaler.pkl")
else:
    model_path = f"./{SAVE_CFG.save_dir}/model.keras"
    scaler_path =f"./{SAVE_CFG.save_dir}/scaler.pkl"

model = load_model(model_path)
scaler=joblib.load(scaler_path)

print("\n\n")
print("########予測の実行結果########")

print(f"{'時間' if MEASURE_CFG.data_axis=='time' else '距離'}軸で実行します")
csv_path= f"./measurement/result/WAVE{MEASURE_CFG.cource.predict:04d}/result_n{'t' if MEASURE_CFG.data_axis=='time' else 'd'}-001.csv" 
data_csv = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
measure_data = data_csv.values.astype(np.float64) # csv用のデータ構造からnumpy配列に変換

result=predict(
    model,
    measure_data,
    scaler,
    RNN_CFG,
    SAVE_CFG.plot_start,
    SAVE_CFG.plot_range,
)

print(result["predict_data"])

save_predict_data(
    run_id,
    result["true_data"],
    result["predict_data"],
    result["rmse_arr"][RNN_CFG.out_steps_num-1],
    result["predict_result_figure"],
    SAVE_CFG
)


print(f"rmse:{result['rmse_arr']}")
print("##############################")
plt.show()
