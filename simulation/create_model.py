import matplotlib.pyplot as plt
from pathlib import Path
from numpy.random import RandomState

from common import create_model
from common import ExperimentsSaver
from function import make_rice_learning_dataset
from configs.config import SIMULATION_CFG,RNN_CFG,SAVE_CFG

rnd=RandomState(0)
dataset,val_dataset,scaler=make_rice_learning_dataset(SIMULATION_CFG,RNN_CFG,rnd)

result=create_model(
    dataset,
    val_dataset,
    RNN_CFG
)

print("\n\n")
print("################モデル作成の実行結果################")
params={**SIMULATION_CFG.model_dump(),**RNN_CFG.model_dump()}
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
    print(f"実行{'id'if SAVE_CFG.use_mlflow else '名'}をrun_id.txtに書き込みました") 
print(f"実行時間:{result['training_time']:.2f}秒")
print("##################################################")

plt.show()