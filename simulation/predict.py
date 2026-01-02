### run_id.txtからrun_id(jsonデータから探すときはrun_name)を読み込んで、そのrunのフォルダからmodel.kerasを取得して予測する ###
### mlrunsフォルダから探すときはUSE_MLFLOWをTrue,exp_runsフォルダから探すときはUSE_MLFLOWをFalseにする ###
import matplotlib.pyplot as plt
from keras.models import load_model

from simulation.setting import RNN_CFG, SAVE_CFG, FADING_CFG
from simulation.simu_func import calc_nakagami_rice_fading, save_predict_data
from common.common_func import predict

# run_idの取得
with open("./simulation/run_id.txt", "r") as f:
    run_id = f.readline().strip()

# run_idで作ったmodelを探す
if SAVE_CFG.use_mlflow:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_path = client.download_artifacts(run_id, "model.keras")
else:
    model_path = f"./{SAVE_CFG.save_dir}/model.keras"

model = load_model(model_path)

# 中上ライスのデータを取得(kerasモデルに渡せるように加工されていない状態)
fading_data = calc_nakagami_rice_fading(FADING_CFG)

# predict関数の中でkerasモデルに渡せるように加工や正規化している
# ???create_model関数には加工してからデータを渡すのに、predict関数には加工前のデータを渡してるの変じゃない?
result = predict(
    model, fading_data, RNN_CFG.input_len, SAVE_CFG.plot_start, SAVE_CFG.plot_range
)

print("\n\n")
print("########予測の実行結果########")

save_predict_data(
    run_id,
    result["true_data"],
    result["predict_data"],
    result["rmse"],
    result["predict_result_figure"],
    SAVE_CFG,
)

print(f"rmse:{result['rmse']:.2f}")
print("##############################")

plt.show()
