from ruamel.yaml import YAML
from itertools import product

from itertools import product

from common.function.function import build_section_grid
from common.schema import RnnConfig,SaveConfig
from simulation.configs.schema import SimulationConfig

yaml=YAML(typ="safe")
with open("simulation/configs/grid_cfg.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)

N_JOBS=cfg['n_jobs'] 

grid_params=cfg['params']

# simulation,model,saveそれぞれで直積
simulation_grid = build_section_grid(grid_params["simulation"])
model_grid   = build_section_grid(grid_params["model"])
save_grid    = build_section_grid(grid_params["save"])

# BaseModelに変換
simulation_grid = [SimulationConfig(**dict(s)) for s in simulation_grid]
model_grid   = [RnnConfig(**dict(m)) for m in model_grid]
save_grid    = [SaveConfig(**dict(s)) for s in save_grid]

# 直積された3つでさらに直積
PARAMS_LIST = [
    {
        "simulation": simu,
        "model": model,
        "save": save,
    }
    for simu, model, save in product(simulation_grid, model_grid, save_grid)
]

print(f"全組み合わせ数: {len(PARAMS_LIST)}")
print(PARAMS_LIST[0])

