from ruamel.yaml import YAML
from itertools import product

from common.schema.config import RnnConfig,SaveConfig
from common.function.func import build_section_grid
from measurement.configs.schema import MeasureConfig
# ???もっと汎用的なコードにしたほうがいい

yaml=YAML(typ="safe")
with open("measurement/configs/grid_cfg.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)

N_JOBS=cfg['n_jobs'] 

grid_params=cfg['params']

# measure,model,saveそれぞれで直積
measure_grid = build_section_grid(grid_params["measure"])
model_grid   = build_section_grid(grid_params["model"])
save_grid    = build_section_grid(grid_params["save"])

# 直積された3つでさらに直積
PARAMS_LIST = [
    {
        "measure": m,
        "model": mdl,
        "save": s,
    }
    for m, mdl, s in product(measure_grid, model_grid, save_grid)
]

# BaseModelに変換
for param in PARAMS_LIST:
    param['measure'] = MeasureConfig(**param['measure']) #type:ignore
    param['model']   = RnnConfig(**param['model']) #type:ignore
    param['save']    = SaveConfig(**param["save"]) #type:ignore


print(f"全組み合わせ数: {len(PARAMS_LIST)}")
print(PARAMS_LIST[0])
