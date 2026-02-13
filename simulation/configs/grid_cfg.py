from ruamel.yaml import YAML
from itertools import product
from pathlib import Path

from common.utils.func import build_section_grid
from common import RnnConfig,SaveConfig

from schema import SimulationConfig

yaml=YAML(typ="safe")
with open(Path(__file__).parent/"grid_cfg.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)

N_JOBS=cfg['n_jobs'] 

grid_params=cfg['params']

# simulation,model,saveそれぞれで直積
simulation_grid = build_section_grid(grid_params["simulation"])
model_grid   = build_section_grid(grid_params["model"])
save_grid    = build_section_grid(grid_params["save"])

# 直積された3つでさらに直積
PARAMS_LIST = [
    {
        "simulation": simu,
        "model": model,
        "save": save,
    }
    for simu, model, save in product(simulation_grid, model_grid, save_grid)
]

# BaseModelに変換
for param in PARAMS_LIST:
    param['simulation'] = SimulationConfig(**param['simulation']) #type:ignore
    param['model']   = RnnConfig(**param['model']) #type:ignore
    param['save']    = SaveConfig(**param["save"]) #type:ignore

print(f"全組み合わせ数: {len(PARAMS_LIST)}")
print(PARAMS_LIST[0])

