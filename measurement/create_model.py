import matplotlib.pyplot as plt
from pathlib import Path

from common import create_model
from common import ExperimentsSaver
from configs.config import MEASURE_CFG,RNN_CFG,SAVE_CFG
from .function import make_learning_dataset

train_dataset,val_dataset,scaler=make_learning_dataset(MEASURE_CFG,RNN_CFG)

result=create_model(
    train_dataset,
    val_dataset,
    RNN_CFG
)

print("\n\n")
print("################モデル作成の実行結果################")

params={
    **MEASURE_CFG.model_dump(), #辞書型に変換
    **RNN_CFG.model_dump()
}
save=ExperimentsSaver(
    model=result['model'],
    params=params,
    metrics={"train_time":result["training_time"]},
    figures={"history":result['history_figure']},
    pkls={"scaler":scaler}
)
run_id=save.save(SAVE_CFG)

with open(Path(__file__).parent/"run_id.txt","w") as f:
    f.write(run_id)
    print(f"実行{'id' if SAVE_CFG.use_mlflow else '名'}をrun_id.txtに書き込みました")

print(f"実行時間:{result['training_time']:.2f}秒")
print(f"{'mlruns' if SAVE_CFG.use_mlflow else SAVE_CFG.base_dir}フォルダに保存しました")
print(f"experiment_name:{SAVE_CFG.experiment_name}")
print(f"run_id(name):{run_id}")
print("##################################################")

plt.show()