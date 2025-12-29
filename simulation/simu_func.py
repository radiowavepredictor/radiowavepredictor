import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array

from simulation.setting import *
from share_func import mw_to_dbm,normalize

def calc_fading():
    theta = np.random.rand(L) * 2 * np.pi
    phi = np.random.rand(L) * 2 * np.pi
    
    fading_data_list=[]
    x=0.0
    for _ in range(DATA_NUM):
        x+=DELTA_D
        fading_data = np.sum(R * np.exp(1j * (theta + (2 * np.pi / LAMBDA_0) * x * np.cos(phi))))
        fading_data /= np.sqrt(L)
        fading_data_list.append(fading_data)
 
    return np.array(fading_data_list)
    
def calc_nakagami_rice_fading(k_rice=K_RICE):
    theta0 = np.random.rand() * 2 * np.pi
    scattered_data_list=calc_fading()
    scattered_data_list = scattered_data_list / np.sqrt(np.mean(np.abs(scattered_data_list)**2))*np.sqrt(1 / (k_rice + 1))
    direct_data_list=[]
    x=0
    for _ in range(DATA_NUM):
        x+=DELTA_D
        direct_data = np.sqrt(k_rice / (k_rice + 1.0)) * np.exp(1j * ((2 * np.pi / LAMBDA_0) * x + theta0))
        direct_data_list.append(direct_data)
    direct_data_list=np.array(direct_data_list)
    
    nakagami_rice_data_list=scattered_data_list+direct_data_list

    return nakagami_rice_data_list
        
### シミュレーション用のデータセット(入力と答え)の配列を用意する関数
def generate_fading_dataset(input_len,data_set_num=DATA_SET_NUM):
    
    fading_data_list_list=[]
    for _ in range(data_set_num):
        fading_data_list_list.append(calc_nakagami_rice_fading())
    fading_data_list_list=np.array(fading_data_list_list)

    power = np.abs(fading_data_list_list) ** 2
    power_db = mw_to_dbm(power)
    data_normalized = normalize(power_db)
    dataset_arr=[]
    for i in range(data_set_num):
        targets = data_normalized[i][input_len:] 

        #datasetの中身はtensorflow特有のオブジェクトで入力と出力(入力に対する答え)のセットが入っている    
        dataset_i=timeseries_dataset_from_array(
            data=data_normalized[i],
            targets=targets,
            sequence_length=input_len,
            batch_size=None,
            shuffle=None
        )
        dataset_arr.append(dataset_i)
    # dataset_arrの中身をすべてつなげる
    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)
    return dataset

### シミュレーションデータセットを訓練用、検証用と用意する関数
def load_fading_data(batch_size,input_len):
    train_dataset = generate_fading_dataset(input_len)
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = generate_fading_dataset(input_len,DATA_SET_NUM//4)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset,val_dataset

### モデル作成時のデータを保存する ###
def save_create_data(
    model,
    history_figure,
    training_time,
    
    experiment_name=EXPERIMENT_NAME,
    run_name=RUN_NAME,
    l=L,
    delta_d=DELTA_D,
    data_num=DATA_NUM,
    data_set_num=DATA_SET_NUM,
    k_rice=K_RICE,
    input_len=INPUT_LEN,
    hidden_nums=HIDDEN_NUMS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    rnn_class=RNN_TYPE,
    optimizer_class=OPTIMIZER_TYPE,
    epochs=EPOCHS,
    use_mlflow=USE_MLFLOW,

    save_dir=SAVE_DIR,
    ):
    
    if use_mlflow:
        print("mlflowに保存します")
        import mlflow
        print("******mlflowのログ******")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("L",l)
            mlflow.log_param("Delta_D",delta_d)
            mlflow.log_param("Data_Num",data_num)
            mlflow.log_param("Data_Set_Num",data_set_num)
            mlflow.log_param("K_Rice",k_rice)

            mlflow.log_param("Input", input_len)
            mlflow.log_param("Layers",len(hidden_nums))
            mlflow.log_param("Units",hidden_nums)
            mlflow.log_param("Batch", batch_size)
            mlflow.log_param("Learning_Rate", learning_rate)
            mlflow.log_param("RNN_Name",rnn_class.__name__)
            mlflow.log_param("Optimizer", optimizer_class.__name__)
            mlflow.log_param("Epochs", epochs)

            mlflow.log_figure(history_figure, "loss_curve.png") 
            mlflow.log_metric("training_time",training_time)
            artifact_dir = mlflow.get_artifact_uri()
            model_path = os.path.join(artifact_dir.replace("file:",""), "model.keras")
            model.save(model_path)
            run_id=run.info.run_id
            with open("./simulation_func/run_id.txt","w") as f:
                f.write(run_id)
        print("************************")
        print("実験名(experiment_name):",experiment_name)
        print("実験id:",run.info.experiment_id)
        print("実行名(run_name):",run_name)
        print("実行id:",run_id)
    else:
        print("jsonで保存します")
        os.makedirs(save_dir, exist_ok=True)

        data = {
            "run_name": run_name,
            "datetime": datetime.now().isoformat(),
            "params": {
                "L": l,
                "Delta_D": delta_d,
                "Data_Num": data_num,
                "Data_Set_Num": data_set_num,
                "K_Rice": k_rice,
                "Input": input_len,
                "Layers": len(hidden_nums),
                "Units": hidden_nums,
                "Batch": batch_size,
                "Learning_Rate": learning_rate,
                "RNN_Name": rnn_class.__name__,
                "Optimizer": optimizer_class.__name__,
                "Epochs": epochs,
            },
            "metrics": {
                "training_time": training_time
            }
        }

        with open(os.path.join(save_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=2)
        history_figure.savefig(os.path.join(save_dir, "loss_curve.png"))
        model.save(os.path.join(save_dir,"model.keras" ))
        with open("./simulation_func/run_id.txt","w") as f:
            f.write(run_name)
        print("experiment_name:",experiment_name)
        print("run_name:",run_name)

def save_predict_data(
    run_id,
    true_data,
    predict_data,
    rmse,
    predict_result_fig,
    use_mlflow=USE_MLFLOW,
    
    base_dir=BASE_DIR,
    experiment_name=EXPERIMENT_NAME
    ):
    
    if use_mlflow:
        print("mlflowに保存します")
        import mlflow
        print("******mlflowのログ******")
        with mlflow.start_run(run_id):
            artifact_dir = mlflow.get_artifact_uri()
            artifact_path = artifact_dir.replace("file:", "")

            true_path = os.path.join(artifact_path, "true.npy")
            pred_path = os.path.join(artifact_path, "predicted.npy")
            np.save(true_path, true_data)
            np.save(pred_path, predict_data)
            mlflow.log_metric("rmse",rmse)
            mlflow.log_figure(predict_result_fig, "predict_results.png") 
        print("************************")
        print("実験名(experiment_name):",experiment_name)
        print("実行id(run_id):",run_id)

    else:
        print("jsonで保存します")
        run_dir=f"./{base_dir}/{experiment_name}/{run_id}"
        with open(f"{run_dir}/data.json", "r") as f:
            data = json.load(f)
        data["metrics"]["rmse"] = rmse

        with open(f"{run_dir}/data.json", "w") as f:
            json.dump(data, f, indent=2)
        predict_result_fig.savefig(f"{run_dir}/predict_results.png")
        np.save(f"{run_dir}/true.npy", true_data)
        np.save(f"{run_dir}/predicted.npy", predict_data)
        print("実験名(experiment_name):",experiment_name)
        print("実験名(run_name(id)):",run_id)