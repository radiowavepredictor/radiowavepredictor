import os
import numpy as np
from ruamel.yaml import YAML
import joblib
from datetime import datetime

from common.function.function import struct_to_flat_dict,to_yaml_safe
from common.schema import RnnConfig,SaveConfig

def save_create_data(
    model,
    scaler,
    history_figure,
    training_time,
    data_cfg,
    rnn_cfg: RnnConfig,
    save_cfg: SaveConfig,
):
    """
    Parameters
    ----------
    data_cfg 
        学習時に使ったデータの設定構造体、もしくは辞書型を期待しています。
    
    Returns
    ----------
    """
    data_params = struct_to_flat_dict(data_cfg)
    data_params = to_yaml_safe(data_params)
    rnn_params = struct_to_flat_dict(rnn_cfg)
    rnn_params = to_yaml_safe(rnn_params)

    if save_cfg.use_mlflow:
        import mlflow

        mlflow.set_experiment(save_cfg.experiment_name)
        with mlflow.start_run(run_name=save_cfg.run_name) as run:
            # measure_cfg を全部保存
            for k, v in data_params.items():
                mlflow.log_param(k, v)

            # rnn_cfg を全部保存
            for k, v in rnn_params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("training_time", training_time)
            mlflow.log_figure(history_figure, "loss_curve.png")

            artifact_dir = mlflow.get_artifact_uri()
            save_path = artifact_dir.replace("file:", "")

            run_id = run.info.run_id
    else:
        os.makedirs(save_cfg.save_dir, exist_ok=True)

        save_path = save_cfg.save_dir
        history_figure.savefig(os.path.join(save_path, "loss_curve.png"))

        run_id = save_cfg.run_name
        
    data = {
        "run_name": save_cfg.run_name,
        "datetime": datetime.now().isoformat(),
        "params": {
            **data_params,
            **rnn_params,
        },
        "metrics": {
            "training_time": training_time,
        },
    }
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)  # インデントの調整
    with open(os.path.join(save_path, "data.yaml"), "w") as f:
        yaml.dump(data, f)

    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
    model.save(os.path.join(save_path, "model.keras"))
    
    return run_id
    
def save_predict_data(
    run_id,
    true_data,
    predict_data,
    rmse,
    predict_result_fig,
    save_cfg: SaveConfig,
):

    if save_cfg.use_mlflow:
        import mlflow

        with mlflow.start_run(run_id):
            artifact_dir = mlflow.get_artifact_uri()
            save_path = artifact_dir.replace("file:", "")

            mlflow.log_metric("rmse", rmse)
            mlflow.log_figure(predict_result_fig, "predict_results.png")

    else:
        save_path = save_cfg.save_dir
        predict_result_fig.savefig(os.path.join(save_path),"predict_results.png")
        
    yaml = YAML(typ="safe")
    yaml.indent(mapping=2, sequence=4, offset=2)  # インデントの調整
    with open(os.path.join(save_path, "data.yaml"), "r") as f:
        data=yaml.load(f)
        
    data["metrics"]["rmse"] = rmse
    data=to_yaml_safe(data)
    
    with open(os.path.join(save_path, "data.yaml"), "w") as f:
        yaml.dump(data, f)
    
    np.save(os.path.join(save_path, "true.npy"), true_data)
    np.save(os.path.join(save_path, "predicted.npy"), predict_data)
