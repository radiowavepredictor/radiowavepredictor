import uuid
from pydantic import BaseModel,Field
from typing import Type
from keras.layers import Layer
from keras.optimizers import Optimizer
from pathlib import Path

from common.registory import RNN_CLASS_MAP,OPTIMIZER_MAP,RNNType,OptimizerType

class RnnConfig(BaseModel):
    model_config={'frozen':True}
    
    rnn_type:RNNType
    optimizer_type:OptimizerType
    in_features:int
    out_steps_num:int
    input_len:int
    hidden_nums:list[int]
    batch_size:int
    epochs:int
    learning_rate:float
    patience:int
    
    @property
    def rnn_class(self) -> Layer:
        return RNN_CLASS_MAP[self.rnn_type.value]
        
    @property
    def optimizer_class(self) -> Type[Optimizer]:
        return OPTIMIZER_MAP[self.optimizer_type.value]

class SaveConfig(BaseModel):
    model_config={'frozen':True}
    
    plot_start:int
    plot_range:int
    
    base_dir:str
    experiment_name:str
    run_name: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    use_mlflow:bool

    recursive_num:int

    @property
    def save_dir(self) -> Path:
        return Path(self.base_dir) / self.experiment_name / self.run_name / "artifacts"

