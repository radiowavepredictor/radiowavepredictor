from pydantic import BaseModel, Field
from pathlib import Path
from ruamel.yaml import YAML
import joblib
import numpy as np
from typing import Any
from urllib.parse import unquote, urlparse

from common.utils.func import flatten_dict, to_yaml_safe
from common.schema.config import SaveConfig


def save_dict_recursively(d: dict, base_path: Path, suffix: str):
    """ネスト辞書を再帰的に保存"""
    for k, v in d.items():
        if isinstance(v, dict):
            save_dict_recursively(v, base_path / k, suffix)
        else:
            base_path.mkdir(parents=True, exist_ok=True)
            if suffix == ".npy":
                np.save(base_path / f"{k}{suffix}", v)
            elif suffix == ".pkl":
                joblib.dump(v, base_path / f"{k}{suffix}")
            elif suffix == ".png":
                v.savefig(base_path / f"{k}{suffix}")
            else:
                raise ValueError(f"Unsupported suffix: {suffix}")

# 保存したいパラメータをプロパティとして持っておいて、saveメソッドで保存する
class ExperimentsSaver(BaseModel):
    model: Any | None = None
    # Field(default_factory=dict)で デフォルト値を空のdictにできる
    params: dict = Field(default_factory=dict)  
    metrics: dict = Field(default_factory=dict)
    figures: dict = Field(default_factory=dict)
    npys: dict = Field(default_factory=dict)
    pkls: dict = Field(default_factory=dict)

    def _save_yaml(self, save_dir: Path, run_name):
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)

        data_yaml_path = save_dir / "data.yaml"

        if data_yaml_path.exists():
            # ファイルが存在する場合は読み込む
            with data_yaml_path.open("r") as f:
                data = yaml.load(f)
        else:
            data = {"run_name": run_name}
            data["params"] = {}
            data["metrics"] = {}

        data["params"].update(flatten_dict(self.params))
        data["metrics"].update(flatten_dict(self.metrics))
        data = to_yaml_safe(data)

        with data_yaml_path.open("w") as f:
            yaml.dump(data, f)

    def _save_mlflow(self, save_cfg: SaveConfig, run_id: str | None):
        import mlflow

        if not run_id:
            mlflow.set_experiment(save_cfg.experiment_name)
            with mlflow.start_run(run_name=save_cfg.run_name) as run:
                run_id = run.info.run_id

        with mlflow.start_run(run_id):
            if self.params:
                mlflow.log_params(flatten_dict(self.params))
            if self.metrics:
                mlflow.log_metrics(flatten_dict(self.metrics))

            ## artifactディレクトリのpathを取得する よくわかってない
            artifact_dir = mlflow.get_artifact_uri()
            if artifact_dir.startswith("file:"):
                artifact_dir = unquote(urlparse(artifact_dir).path)
                if (
                    len(artifact_dir) >= 3
                    and artifact_dir[0] == "/"
                    and artifact_dir[2] == ":"
                ):
                    artifact_dir = artifact_dir[1:]
            ##
        return Path(artifact_dir), run_id

    def save(self, save_cfg: SaveConfig, run_id: str | None = None) -> str:
        if save_cfg.use_mlflow:
            artifacts_dir, run_id = self._save_mlflow(save_cfg, run_id)
        else:
            artifacts_dir = save_cfg.artifacts_dir
            run_id = save_cfg.run_name

        save_dict_recursively(self.figures, artifacts_dir, ".png")
        save_dict_recursively(self.npys, artifacts_dir, ".npy")
        save_dict_recursively(self.pkls, artifacts_dir, ".pkl")

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save(artifacts_dir / "model.keras")

        self._save_yaml(artifacts_dir, save_cfg.run_name)

        return run_id  # type:ignore[arg-type] run_idがstr型であることは保証したのでignore