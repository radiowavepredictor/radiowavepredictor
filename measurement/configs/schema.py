from pydantic import BaseModel
from typing import Literal

class CourceConfig(BaseModel):
    model_config={'frozen':True}
    
    train:list[int]
    val:list[int]
    predict:int

class MeasureConfig(BaseModel):
    model_config={'frozen':True}
    
    train_cources:list[int]
    val_cources:list[int]
    predict_cource:int
    start_ratio:float
    end_ratio:float
    data_axis:Literal["time","distance"]

