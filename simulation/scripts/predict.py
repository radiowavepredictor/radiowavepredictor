### run_id.txtからrun_id(jsonデータから探すときはrun_name)を読み込んで、そのrunのフォルダからmodel.kerasを取得して予測する ###
### mlrunsフォルダから探すときはUSE_MLFLOWをTrue,exp_runsフォルダから探すときはUSE_MLFLOWをFalseにする ###
import matplotlib.pyplot as plt
from keras.models import load_model

from simulation.configs.config import RNN_CFG, SAVE_CFG, FADING_CFG
from simulation.simu_func import evaluate_model, save_predict_data

# run_idの取得
with open("./simulation/scripts/run_id.txt", "r") as f:
    run_id = f.readline().strip()

# run_idで作ったmodelを探す
if SAVE_CFG.use_mlflow:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_path = client.download_artifacts(run_id, "model.keras")
else:
    model_path = f"./{SAVE_CFG.save_dir}/model.keras"

model = load_model(model_path)


print("\n\n")
print("########予測の実行結果########")

# 中で複数回predictしてる
first_result,rmse_mean=evaluate_model(model,FADING_CFG,RNN_CFG,SAVE_CFG)

save_predict_data(
    run_id,
    first_result["true_data"],
    first_result["predict_data"],
    first_result["rmse"],
    first_result["predict_result_figure"],
    rmse_mean,
    SAVE_CFG,
)

print(f"rmseの平均:{rmse_mean:.2f}")
print("##############################")

plt.show()
