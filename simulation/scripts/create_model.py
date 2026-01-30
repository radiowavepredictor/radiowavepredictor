import matplotlib.pyplot as plt

from common.function.model import create_model
from common.function.save import save_create_data
from simulation.function import load_fading_dataset
from simulation.configs.config import SIMULATION_CFG,RNN_CFG,SAVE_CFG

(dataset,val_dataset),scaler=load_fading_dataset(SIMULATION_CFG,RNN_CFG)

result=create_model(
    dataset,
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
    SIMULATION_CFG,
    RNN_CFG,
    SAVE_CFG
)

with open("./simulation/scripts/run_id.txt","w") as f:
    f.write(run_id)
    print(f"実行{'id'if SAVE_CFG.use_mlflow else '名'}をrun_id.txtに書き込みました") 
print(f"実行時間:{result['training_time']:.2f}秒")
print("##################################################")

plt.show()