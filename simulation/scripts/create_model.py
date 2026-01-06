import matplotlib.pyplot as plt

from simulation.simu_func import load_fading_data,save_create_data
from simulation.configs.config import FADING_CFG,RNN_CFG,SAVE_CFG
from common.common_func import create_model

dataset,val_dataset=load_fading_data(FADING_CFG,RNN_CFG)

result=create_model(
    dataset,
    val_dataset,
    RNN_CFG.input_len,
    RNN_CFG.in_features,
    RNN_CFG.hidden_nums,
    RNN_CFG.rnn_class,
    RNN_CFG.optimizer_class,
    RNN_CFG.out_steps_num,
    RNN_CFG.learning_rate,
    RNN_CFG.epochs
)

print("\n\n")
print("################モデル作成の実行結果################")

run_id=save_create_data(
    result['model'],
    result['history_figure'],
    result['training_time'],
    SAVE_CFG,
    FADING_CFG,
    RNN_CFG
)

with open("./simulation/scripts/run_id.txt","w") as f:
    f.write(run_id)
    print(f"実行{SAVE_CFG.use_mlflow if "id" else "名"}をrun_id.txtに書き込みました") #???ここ多分おかしい
print(f"実行時間:{result['training_time']:.2f}秒")
print("##################################################")

plt.show()