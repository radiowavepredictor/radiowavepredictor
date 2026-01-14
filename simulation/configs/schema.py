from pydantic import BaseModel

class SimulationConfig(BaseModel):
    model_config={'frozen':True}
    
    data_num: int
    data_set_num: int
    l: int
    delta_d: float
    c:float
    f:float
    r: float
    k_rice: float
    predicted_dataset_num:int

    @property
    def lambda_0(self):
        return self.c/self.f
