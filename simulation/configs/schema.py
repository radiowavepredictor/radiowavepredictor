from .fading_schema import RiceConfig

class SimulationConfig(RiceConfig):
    model_config={'frozen':True}
    
    data_set_num: int
    predicted_dataset_num:int
    seed: int
    target_k_db: float=10.0
    sigma_shadow:float
    d_c:float