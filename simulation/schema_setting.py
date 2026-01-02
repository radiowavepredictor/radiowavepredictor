from typing import Type
from dataclasses import dataclass
from keras.layers import Layer
from keras.optimizers import Optimizer

@dataclass(frozen=True)
class FadingConfig:
    data_num: int
    data_set_num: int
    l: int
    delta_d: float
    lambda_0: float
    r: float
    k_rice: float

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
    
    experiment_name:str
    run_name:str
    use_mlflow:bool
    save_dir:str
    predicted_dataset_num:int
    recursive_num:int