from pydantic import BaseModel
import numpy as np

class SimulationConfig(BaseModel):
    model_config={'frozen':True}
    
    data_num: int
    data_set_num: int
    l: int
    delta_d: float
    c:float
    f:float
    target_k_db: float
    predicted_dataset_num:int

    @property
    def lambda_0(self):
        return self.c/self.f

    @property
    def r0(self):
        K_linear = 10 ** (self.target_k_db / 10.0)
        r0 = np.sqrt(K_linear)
        return r0