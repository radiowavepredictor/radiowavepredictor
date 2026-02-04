import matplotlib.pyplot as plt
from pathlib import Path

from common.function.model import create_model
from common.function.save_class import SaveClass
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
params={**SIMULATION_CFG.model_dump(),**RNN_CFG.model_dump()}
save=SaveClass(
    model=result['model'],
    params=params,
    metrics={"train_time":result["training_time"]},
    figures={"history":result['history_figure']},
    pkls={"scaler":scaler}
)
run_id=save.save(SAVE_CFG)
'''
run_id=save_create_data(
    result['model'],
    scaler,
    result['history_figure'],
    result['training_time'],
    SIMULATION_CFG,
    RNN_CFG,
    SAVE_CFG
)
'''
with open(Path("simulation")/"scripts"/"run_id.txt","w") as f:
    f.write(run_id)
    print(f"実行{'id'if SAVE_CFG.use_mlflow else '名'}をrun_id.txtに書き込みました") 
print(f"実行時間:{result['training_time']:.2f}秒")
print("##################################################")

plt.show()