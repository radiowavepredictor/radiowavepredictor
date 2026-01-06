import uuid
from typing import Type,Literal
from dataclasses import dataclass,field
from keras.layers import Layer
from keras.optimizers import Optimizer
from pathlib import Path

@dataclass(frozen=True)
class MeasureConfig:
    train_corces:list[int]
    val_corces:list[int]
    predict_corce:int
    start_ratio:float
    end_ratio:float
    data_axis:Literal["time","distance"]

@dataclass(frozen=True)
class RnnConfig:
    rnn_class:Type[Layer]
    optimizer_class:Type[Optimizer]
    in_features:int
    out_steps_num:int
    input_len:int
    hidden_nums:list[int]
    batch_size:int
    epochs:int
    learning_rate:float

@dataclass(frozen=True)
class SaveConfig:
    plot_start:int
    plot_range:int
    
    base_dir:str
    experiment_name:str
    run_name:str = field(init=False)
    use_mlflow:bool

    recursive_num:int

    def __post_init__(self):
        object.__setattr__(self, "run_name",str(uuid.uuid4().hex[:8])) #run_nameはランダムで被らない名前に設定する

    @property
    def save_dir(self) -> Path:
        return Path(self.base_dir) / self.experiment_name / self.run_name