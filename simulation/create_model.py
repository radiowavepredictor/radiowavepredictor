import matplotlib.pyplot as plt

from simulation.simu_func import load_fading_data,save_create_data
from simulation.setting import *
from share_func import create_model

dataset,val_dataset=load_fading_data(BATCH_SIZE,INPUT_LEN)

result=create_model(
    dataset,
    val_dataset,
    INPUT_LEN,
    IN_FEATURES,
    HIDDEN_NUMS,
    RNN_TYPE,
    OPTIMIZER_TYPE,
    OUT_STEPS_NUM,
    LEARNING_RATE,
    EPOCHS
)

print("\n\n")
print("######モデル作成の実行結果######")

save_create_data(
    result['model'],
    result['history_figure'],
    result['training_time']
)

print(f"実行時間:{result['training_time']:.2f}秒")
print("##############################")

plt.show()