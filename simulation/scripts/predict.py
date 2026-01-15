### run_id.txtからrun_id(jsonデータから探すときはrun_name)を読み込んで、そのrunのフォルダからmodel.kerasを取得して予測する ###
### mlrunsフォルダから探すときはUSE_MLFLOWをTrue,exp_runsフォルダから探すときはUSE_MLFLOWをFalseにする ###
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib

from common.function.save import save_predict_data
from simulation.configs.config import RNN_CFG, SAVE_CFG, SIMULATION_CFG
from simulation.function import evaluate_model, wrap_save_predict_data

# run_idの取得
with open("./simulation/scripts/run_id.txt", "r") as f:
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
scaler = joblib.load(scaler_path)


print("\n\n")
print("########予測の実行結果########")

# 中で複数回predictしてる
first_result,rmse_mean_arr=evaluate_model(model,scaler,SIMULATION_CFG,RNN_CFG,SAVE_CFG)

wrap_save_predict_data(
    run_id,
    first_result["true_data"],
    first_result["predict_data"],
    first_result["rmse_arr"][RNN_CFG.out_steps_num-1],
    rmse_mean_arr[RNN_CFG.out_steps_num-1],
    first_result["predict_result_figure"],
    SAVE_CFG,
)

print(f"rmse:{rmse_mean_arr}")
print(f"rmseの平均:{rmse_mean_arr[RNN_CFG.out_steps_num-1]:.2f}")
print("##############################")

plt.show()
