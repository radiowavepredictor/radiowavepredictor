import matplotlib.pyplot as plt
import time 

from common.function.function import create_model
from common.function.save import save_create_data
from measurement.configs.config import MEASURE_CFG,RNN_CFG,SAVE_CFG
from measurement.function import load_learning_dataset

#コード実行時間計測
start_time=time.time()

(train_dataset,val_dataset),scaler=load_learning_dataset(MEASURE_CFG,RNN_CFG)

end_time=time.time()
print(f"実行時間:{(end_time-start_time):2f}秒")

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

with open("./measurement/scripts/run_id.txt","w") as f:
    f.write(run_id)
    print(f"実行{'id' if SAVE_CFG.use_mlflow else '名'}をrun_id.txtに書き込みました")

print(f"実行時間:{result['training_time']:.2f}秒")
print(f"{'mlruns' if SAVE_CFG.use_mlflow else SAVE_CFG.base_dir}に保存しました")
print(f"experiment_name:{SAVE_CFG.experiment_name}")
print(f"run_id(name):{run_id}")
print("##################################################")

plt.show()