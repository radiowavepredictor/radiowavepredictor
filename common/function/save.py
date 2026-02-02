import numpy as np
from ruamel.yaml import YAML
import joblib
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote,urlparse

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
            # data_cfg を保存
            for k, v in data_params.items():
                mlflow.log_param(k, v)

            # rnn_cfg を保存
            for k, v in rnn_params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("training_time", training_time)
            mlflow.log_figure(history_figure, "loss_curve.png")

            artifact_dir = mlflow.get_artifact_uri()
            if artifact_dir.startswith("file:"):
                save_path=unquote(urlparse(artifact_dir).path)
                if len(save_path)>=3 and save_path[0]=="/" and save_path[2]==":":
                    save_path=save_path[1:]
            else:
                save_path=artifact_dir

            run_id = run.info.run_id
    else:
        save_path = save_cfg.save_dir
        save_path.mkdir(parents=True, exist_ok=True)

        history_figure.savefig(save_path / "loss_curve.png")
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

    save_dir=Path(save_path)
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    data_yaml_path = save_dir / "data.yaml"
    with data_yaml_path.open("w") as f:
        yaml.dump(data, f)

    joblib.dump(scaler, save_dir / "scaler.pkl")
    model.save(save_dir / "model.keras")

    return run_id

def save_predict_data(
    run_id,
    true_data,
    predict_data_dict,
    predict_time,
    rmse_dict,
    predict_fig_dict,
    save_cfg: SaveConfig,
):
    if save_cfg.use_mlflow:
        import mlflow

        with mlflow.start_run(run_id):
            artifact_dir = mlflow.get_artifact_uri()
            if artifact_dir.startswith("file:"):
                save_path=unquote(urlparse(artifact_dir).path)
                if len(save_path)>=3 and save_path[0]=="/" and save_path[2]==":":
                    save_path=save_path[1:]
            else:
                save_path=artifact_dir
            for key,value in rmse_dict.items():
                mlflow.log_metric(key,value)
            mlflow.log_metric("predict_time",predict_time)
            for key, value in predict_fig_dict.items():
                mlflow.log_figure(
                    value,
                    artifact_file=f"predict_fig/{key}.png"
                )

    else:
        save_path = save_cfg.save_dir
        save_path.mkdir(parents=True, exist_ok=True)
        for key, value in predict_fig_dict.items():
            value.savefig(save_path/"predict_fig"/f"{key}.png")

    save_dir=Path(save_path)
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.default_flow_style = False 

    data_yaml_path = save_dir / "data.yaml"

    with data_yaml_path.open("r") as f:
        data = yaml.load(f)
        
    data["metrics"]["rmse"]={}
    for key,value in rmse_dict.items():
        data["metrics"]["rmse"][key]=value

    data["metrics"]["predict_time"] = predict_time
    data = to_yaml_safe(data)
    
    with data_yaml_path.open("w") as f:
        yaml.dump(data, f)
    np.save(save_dir / "true.npy", true_data)
    (save_dir/"predict_data").mkdir(parents=True, exist_ok=True)
    for key,value in predict_data_dict.items():
        np.save(save_dir / "predict_data"/ f"{key}.npy",value)
