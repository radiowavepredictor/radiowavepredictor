import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from pathlib import Path

from common import predict
from common import ExperimentsSaver
from configs.config import RNN_CFG,SAVE_CFG,MEASURE_CFG

# run_idの取得
with open(Path(__file__).parent/"run_id.txt", "r") as f:
    run_id = f.readline().strip()

# run_idで作ったmodelを探す
if SAVE_CFG.use_mlflow:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_path = client.download_artifacts(run_id, "model.keras")
    scaler_path = client.download_artifacts(run_id,"scaler.pkl")
else:
    model_path = Path(f"{SAVE_CFG.artifacts_dir}")/"model.keras"
    scaler_path =Path(f"{SAVE_CFG.artifacts_dir}")/"scaler.pkl"

model = load_model(model_path)
scaler=joblib.load(scaler_path)

print("\n\n")
print("########予測の実行結果########")

print(f"{'時間' if MEASURE_CFG.data_axis=='time' else '距離'}軸で実行します")
csv_path= Path("measurement")/"result"/f"WAVE{MEASURE_CFG.cource.predict:04d}"/f"result_n{'t' if MEASURE_CFG.data_axis=='time' else 'd'}-001.csv" 
data_csv = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
measure_data = data_csv.values.astype(np.float64) # csv用のデータ構造からnumpy配列に変換

result=predict(
    model,
    measure_data,
    scaler,
    RNN_CFG,
    SAVE_CFG.plot_start,
    SAVE_CFG.plot_range,
    MEASURE_CFG.sampling_rate
)

print(result.predict_data)

save=ExperimentsSaver(
    metrics={**result.rmse,"predict_time":result.predict_time},
    figures={"predict_figure":result.predict_figure},
    npys={"true_data":result.true_data,"predict_data":result.predict_data}
)
save.save(SAVE_CFG,run_id)

print(f"rmse:{result.rmse}")
print("##############################")
plt.show()
