### mlflowではなくjson形式で保存したデータをmlflowのデータに合体させる ###
### mlrunsフォルダとexp_runsフォルダ(名前変えてるかも)両方をRadioWavePredictフォルダの直下において実行する ###
### EXPERIMENT_NAME(実験名)が適切か、合体するデータ同士で保存してるデータの内容が全然違わないか、などを確認してから実行する ###
import json
import mlflow
from pathlib import Path
from simulation_func.simu_func import *

experiment_dir=Path(f"{BASE_DIR}/{EXPERIMENT_NAME}")

mlflow.set_experiment(EXPERIMENT_NAME)

for run_dir in experiment_dir.iterdir():
    with open(f"{run_dir}/data.json") as f:
        data = json.load(f)

    run_name = data.get("run_name", run_dir.name)
    params = data.get("params", {})
    metrics = data.get("metrics", {})
    datetime = data.get("datetime", None)

    with mlflow.start_run(run_name=run_name):

        for key, value in params.items():
            if isinstance(value, list):
                mlflow.log_param(key, ",".join(map(str, value)))
            else:
                mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        if datetime:
            mlflow.set_tag("datetime", datetime)

        for item in run_dir.iterdir():
            if item.name != "data.json":
                mlflow.log_artifact(item)
