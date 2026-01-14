from pydantic import BaseModel
from typing import Literal

class CourceConfig(BaseModel):
    model_config={'frozen':True}
    
    train:list[int]
    val:list[int]
    predict:int

class MeasureConfig(BaseModel):
    model_config={'frozen':True}
    
    cource:CourceConfig
    start_ratio:float
    end_ratio:float
    data_axis:Literal["time","distance"]

