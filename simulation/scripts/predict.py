### run_id.txtからrun_id(jsonデータから探すときはrun_name)を読み込んで、そのrunのフォルダからmodel.kerasを取得して予測する ###
### mlrunsフォルダから探すときはUSE_MLFLOWをTrue,exp_runsフォルダから探すときはUSE_MLFLOWをFalseにする ###
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
from pathlib import Path

from simulation.configs.config import RNN_CFG, SAVE_CFG, SIMULATION_CFG
from simulation.function import evaluate_model
from common.function.save_class import SaveClass

# run_idの取得
with open(Path("simulation")/"scripts"/"run_id.txt", "r") as f:
    run_id = f.readline().strip()
    
# run_idで作ったmodelを探す
if SAVE_CFG.use_mlflow:
    from mlflow.tracking import MlflowClient
    
    save_cfg = SAVE_CFG.model_copy(update={"run_name": "run_name"})
    client = MlflowClient()
    model_path = client.download_artifacts(run_id, "model.keras")
    scaler_path = client.download_artifacts(run_id,"scaler.pkl")
else:
    save_cfg=SAVE_CFG.model_copy(update={"run_name":run_id})
    model_path = Path(save_cfg.save_dir)/"model.keras" 
    scaler_path =Path(save_cfg.save_dir)/"scaler.pkl"

model = load_model(model_path)
scaler = joblib.load(scaler_path)

print("\n\n")
print("########予測の実行結果########")

# 中で複数回predictしてる
first_result,rmse_mean_dict=evaluate_model(model,scaler,SIMULATION_CFG,RNN_CFG,save_cfg)

save=SaveClass(
    metrics={**first_result.rmse,"predict_time":first_result.predict_time,**rmse_mean_dict},
    figures={"predict_figure":first_result.predict_figure},
    npys={"true_data":first_result.true_data,"predict_data":first_result.predict_data}
)
save.save(save_cfg,run_id)
'''
wrap_save_predict_data(
    run_id,
    first_result.true_data,
    first_result.predict_data,
    first_result.predict_time,
    first_result.rmse,
    rmse_mean_dict,
    first_result.predict_figure,
    save_cfg,
)
'''
print(first_result.rmse)
print(f"rmse:{rmse_mean_dict}")
print("##############################")

plt.show()
