import os
from datetime import datetime
import numpy as np
import json
from dataclasses import replace
import tensorflow as tf
from keras.utils import timeseries_dataset_from_array

from simulation.schema_setting import SaveConfig, FadingConfig, RnnConfig
from common.common_func import mw_to_dbm, normalize, predict


def calc_fading(fading_cfg: FadingConfig):
    theta = np.random.rand(fading_cfg.l) * 2 * np.pi
    phi = np.random.rand(fading_cfg.l) * 2 * np.pi

    fading_data_list = []
    x = 0.0
    for _ in range(fading_cfg.data_num):
        x += fading_cfg.delta_d
        fading_data = np.sum(
            fading_cfg.r
            * np.exp(1j * (theta + (2 * np.pi / fading_cfg.lambda_0) * x * np.cos(phi)))
        )
        fading_data /= np.sqrt(fading_cfg.l)
        fading_data_list.append(fading_data)

    return np.array(fading_data_list)


def calc_nakagami_rice_fading(fading_cfg: FadingConfig):
    theta0 = np.random.rand() * 2 * np.pi
    scattered_data_list = calc_fading(fading_cfg)
    scattered_data_list = (
        scattered_data_list
        / np.sqrt(np.mean(np.abs(scattered_data_list) ** 2))
        * np.sqrt(1 / (fading_cfg.k_rice + 1))
    )
    direct_data_list = []
    x = 0
    for _ in range(fading_cfg.data_num):
        x += fading_cfg.delta_d
        direct_data = np.sqrt(fading_cfg.k_rice / (fading_cfg.k_rice + 1.0)) * np.exp(
            1j * ((2 * np.pi / fading_cfg.lambda_0) * x + theta0)
        )
        direct_data_list.append(direct_data)
    direct_data_list = np.array(direct_data_list)

    nakagami_rice_data_list = scattered_data_list + direct_data_list

    return nakagami_rice_data_list


### シミュレーション用のデータセット(入力と答え)をdata_set_num分用意する関数
def generate_fading_dataset_list(fading_cfg: FadingConfig, rnn_cfg: RnnConfig):
    fading_data_list_list = []
    for _ in range(fading_cfg.data_set_num):
        fading_data_list_list.append(calc_nakagami_rice_fading(fading_cfg))
    fading_data_list_list = np.array(fading_data_list_list)

    power = np.abs(fading_data_list_list) ** 2
    power_db = mw_to_dbm(power)
    data_normalized = normalize(power_db)
    dataset_arr = []
    for i in range(fading_cfg.data_set_num):
        targets = data_normalized[i][rnn_cfg.input_len :]
        # datasetの中身はtensorflow特有のオブジェクトで入力と出力(入力に対する答え)のセットが入っている
        dataset_i = timeseries_dataset_from_array(
            data=data_normalized[i],
            targets=targets,
            sequence_length=rnn_cfg.input_len,
            batch_size=None,  # type:ignore[arg-type]
            shuffle=False,
        )
        dataset_arr.append(dataset_i)
    # dataset_arrの中身をすべてつなげる
    dataset = dataset_arr[0]
    for ds in dataset_arr[1:]:
        dataset = dataset.concatenate(ds)
    return dataset


### シミュレーションデータセットを訓練用、検証用と用意する関数
def load_fading_data(fading_cfg: FadingConfig, rnn_cfg: RnnConfig):
    train_dataset = generate_fading_dataset_list(fading_cfg, rnn_cfg)
    train_dataset = (
        train_dataset.shuffle(buffer_size=10000)
        .batch(rnn_cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_fading_cfg = replace(fading_cfg, data_set_num=fading_cfg.data_set_num // 4)
    val_dataset = generate_fading_dataset_list(val_fading_cfg, rnn_cfg)
    val_dataset = val_dataset.batch(rnn_cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset


### 複数のデータセットで予測を行いrmseの平均を算出する 1回目のデータセットだけ詳細な情報を返す
def evaluate_model(
    model, fading_cfg: FadingConfig, rnn_cfg: RnnConfig, save_cfg: SaveConfig
):
    # 中上ライスのデータを取得(kerasモデルに渡せるように加工されていない状態)
    fading_data = calc_nakagami_rice_fading(fading_cfg)

    # predict関数の中でkerasモデルに渡せるように加工したり正規化などをしている
    # ???create_model関数には加工してからデータを渡すのに、predict関数には加工前のデータを渡してるの変じゃない?
    first_result = predict(
        model, fading_data, rnn_cfg.input_len, save_cfg.plot_start, save_cfg.plot_range
    )

    rmse_sum = first_result["rmse"]
    predict_num = save_cfg.predicted_dataset_num

    for _ in range(predict_num - 1):  # ↑で一回実行してるのでその分減らす
        fading_data = calc_nakagami_rice_fading(fading_cfg)

        result_i = predict(
            model,fading_data,rnn_cfg.input_len,save_cfg.plot_start,save_cfg.plot_range,
        )
        rmse_sum += result_i["rmse"]

    rmse_mean = rmse_sum / predict_num
    return first_result,rmse_mean


### モデル作成時のデータを保存する ###
def save_create_data(
    model,
    history_figure,
    training_time,
    save_cfg: SaveConfig,
    fading_cfg: FadingConfig,
    rnn_cfg: RnnConfig,
):

    if save_cfg.use_mlflow:
        print("mlflowに保存します")
        import mlflow

        print("******mlflowのログ******")
        mlflow.set_experiment(save_cfg.experiment_name)
        with mlflow.start_run(run_name=save_cfg.run_name) as run:
            mlflow.log_param("L", fading_cfg.l)
            mlflow.log_param("Delta_D", fading_cfg.delta_d)
            mlflow.log_param("Data_Num", fading_cfg.data_num)
            mlflow.log_param("Data_Set_Num", fading_cfg.data_set_num)
            mlflow.log_param("K_Rice", fading_cfg.k_rice)

            mlflow.log_param("Input", rnn_cfg.input_len)
            mlflow.log_param("Layers", len(rnn_cfg.hidden_nums))
            mlflow.log_param("Units", rnn_cfg.hidden_nums)
            mlflow.log_param("Batch", rnn_cfg.batch_size)
            mlflow.log_param("Learning_Rate", rnn_cfg.learning_rate)
            mlflow.log_param("RNN_Name", rnn_cfg.rnn_class.__name__)
            mlflow.log_param("Optimizer", rnn_cfg.optimizer_class.__name__)
            mlflow.log_param("Epochs", rnn_cfg.epochs)

            mlflow.log_figure(history_figure, "loss_curve.png")
            mlflow.log_metric("Training_Time", training_time)
            artifact_dir = mlflow.get_artifact_uri()
            model_path = os.path.join(artifact_dir.replace("file:", ""), "model.keras")
            model.save(model_path)
            run_id = run.info.run_id
        print("************************")
        print("実験名(experiment_name):", save_cfg.experiment_name)
        print("実験id:", run.info.experiment_id)
        print("実行名(run_name):", save_cfg.run_name)
        print("実行id:", run_id)
        return run_id
    else:
        print("jsonで保存します")
        os.makedirs(save_cfg.save_dir, exist_ok=True)

        data = {
            "run_name": save_cfg.run_name,
            "datetime": datetime.now().isoformat(),
            "params": {
                "L": fading_cfg.l,
                "Delta_D": fading_cfg.delta_d,
                "Data_Num": fading_cfg.data_num,
                "Data_Set_Num": fading_cfg.data_set_num,
                "K_Rice": fading_cfg.k_rice,
                "Input": rnn_cfg.input_len,
                "Layers": len(rnn_cfg.hidden_nums),
                "Units": rnn_cfg.hidden_nums,
                "Batch": rnn_cfg.batch_size,
                "Learning_Rate": rnn_cfg.learning_rate,
                "RNN_Name": rnn_cfg.rnn_class.__name__,
                "Optimizer": rnn_cfg.optimizer_class.__name__,
                "Epochs": rnn_cfg.epochs,
            },
            "metrics": {"Training_Time": training_time},
        }
        save_dir = save_cfg.save_dir
        with open(os.path.join(save_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=2)
        history_figure.savefig(os.path.join(save_dir, "loss_curve.png"))
        model.save(os.path.join(save_dir, "model.keras"))
        print("experiment_name:", save_cfg.experiment_name)
        print("run_name:", save_cfg.run_name)
        return save_cfg.run_name


def save_predict_data(
    run_id,
    true_data,
    predict_data,
    rmse,
    predict_result_fig,
    rmse_mean,
    save_cfg: SaveConfig,
):

    if save_cfg.use_mlflow:
        print("mlflowに保存します")
        import mlflow

        with mlflow.start_run(run_id):
            artifact_dir = mlflow.get_artifact_uri()
            artifact_path = artifact_dir.replace("file:", "")

            true_path = os.path.join(artifact_path, "true.npy")
            pred_path = os.path.join(artifact_path, "predicted.npy")
            np.save(true_path, true_data)
            np.save(pred_path, predict_data)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("RMSE_MEAN", rmse_mean)
            mlflow.log_figure(predict_result_fig, "predict_results.png")
        print("実験名(experiment_name):", save_cfg.experiment_name)
        print("実行id(run_id):", run_id)

    else:
        save_dir = save_cfg.save_dir
        print("jsonで保存します")
        with open(f"{save_dir}/data.json", "r") as f:
            data = json.load(f)
        data["metrics"]["RMSE"] = rmse
        data["metrics"]["RMSE_MEAN"] = rmse_mean

        with open(f"{save_dir}/data.json", "w") as f:
            json.dump(data, f, indent=2)
        predict_result_fig.savefig(f"{save_dir}/predict_results.png")
        np.save(f"{save_dir}/true.npy", true_data)
        np.save(f"{save_dir}/predicted.npy", predict_data)
        print("実験名(experiment_name):", save_cfg.experiment_name)
        print("実行名(run_id(name)):", run_id)
