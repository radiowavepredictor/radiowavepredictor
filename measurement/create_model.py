import matplotlib.pyplot as plt
from pathlib import Path

from common.function.model import create_model
from common.function.save import save_create_data
from measurement.configs.config import MEASURE_CFG,RNN_CFG,SAVE_CFG
from measurement.function import load_learning_dataset


(train_dataset,val_dataset),scaler=load_learning_dataset(MEASURE_CFG,RNN_CFG)

result=create_model(
    train_dataset,
    val_dataset,
    RNN_CFG
)

print("\n\n")
print("################モデル作成の実行結果################")

run_id=save_create_data(
    result['model'],
    scaler,
    result['history_figure'],
    result['training_time'],
    MEASURE_CFG,
    RNN_CFG,
    SAVE_CFG
)

with open(Path("measurement")/"scripts"/"run_id.txt","w") as f:
    f.write(run_id)
    print(f"実行{'id' if SAVE_CFG.use_mlflow else '名'}をrun_id.txtに書き込みました")

print(f"実行時間:{result['training_time']:.2f}秒")
print(f"{'mlruns' if SAVE_CFG.use_mlflow else SAVE_CFG.base_dir}に保存しました")
print(f"experiment_name:{SAVE_CFG.experiment_name}")
print(f"run_id(name):{run_id}")
print("##################################################")

plt.show()