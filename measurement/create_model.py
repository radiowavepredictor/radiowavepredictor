import matplotlib.pyplot as plt
import time 

from measurement.configs.measure_cfg import MEASURE_CFG,RNN_CFG,SAVE_CFG
from measurement.measure_func import load_training_data,save_create_data
from common.common_func import create_model

#コード実行時間計測
start_time=time.time()

train_dataset,val_dataset,mean,std=load_training_data(MEASURE_CFG,RNN_CFG)

end_time=time.time()
print(f"実行時間:{(end_time-start_time):2f}秒")

result=create_model(
    train_dataset,
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
    MEASURE_CFG,
    RNN_CFG,
    SAVE_CFG
)

with open("./measurement/scripts/run_id.txt","w") as f:
    f.write(run_id)
    print(f"実行{"id" if SAVE_CFG.use_mlflow else "名"}をrun_id.txtに書き込みました")
with open("./measurement/scripts/mean.txt","w") as f:
    f.write(str(mean))
with open("./measurement/scripts/std.txt","w") as f:
    f.write(str(std))

print(f"実行時間:{result['training_time']:.2f}秒")
print("##################################################")

plt.show()