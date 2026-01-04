import uuid
from typing import Type
from dataclasses import dataclass,field
from keras.layers import Layer
from keras.optimizers import Optimizer
from pathlib import Path

@dataclass(frozen=True)
class FadingConfig:
    data_num: int
    data_set_num: int
    l: int
    delta_d: float
    c:float
    f:float
    r: float
    k_rice: float
    lambda_0: float = field(init=False) 

    def __post_init__(self):
        object.__setattr__(self, "lambda_0", float(self.c) / float(self.f))

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

    predicted_dataset_num:int
    recursive_num:int

    def __post_init__(self):
        object.__setattr__(self, "run_name",str(uuid.uuid4().hex[:8])) #run_nameはランダムで被らない名前に設定する

    @property
    def save_dir(self) -> Path:
        return Path(self.base_dir) / self.experiment_name / self.run_name