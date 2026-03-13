from fading_schema import RiceConfig

class SimulationConfig(RiceConfig):
    model_config={'frozen':True}
    
    data_set_num: int
    predicted_dataset_num:int
    seed: int