### experimentのフォルダをmlflowのexperimentに合体させる ###
### mlrunsフォルダとexp_runsフォルダ(名前変えてるかも)両方をRadioWavePredictフォルダの直下において実行する ###
### EXPERIMENT_NAME(実験名)が適切か、合体するデータ同士で保存してるデータの内容が全然違わないか、などを確認してから実行する ###
import mlflow
from pathlib import Path
from ruamel.yaml import YAML


MERGE_EXPERIMENT= Path("test")
MLFLOW_EXPERIMENT="testtesttest"

mlflow.set_experiment(MLFLOW_EXPERIMENT)

for run_dir in MERGE_EXPERIMENT.iterdir():
    yaml=YAML(typ="safe")
    yaml.indent(mapping=2, sequence=4, offset=2)  # インデントの調整
    
    artifacts_dir=run_dir/"artifacts"
    if artifacts_dir.is_dir():
        yaml_path=artifacts_dir/"data.yaml"
    else:
        yaml_path=run_dir/"data.yaml"
        
    with yaml_path.open( "r") as f:
        data=yaml.load(f)


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
            mlflow.log_artifact(str(item))
