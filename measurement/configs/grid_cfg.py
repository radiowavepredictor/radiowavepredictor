from ruamel.yaml import YAML
from itertools import product

from itertools import product
from copy import deepcopy

from common.schema import RnnConfig,SaveConfig
from measurement.configs.schema import MeasureConfig
# ???もっと汎用的なコードにしたほうがいい

# 辞書型からを最初の(1段目の?)階層で直積する
def build_section_grid(section: dict):
    fixed = {}
    grid_keys = []
    grid_values = []

    for k, v in section.items():
        if isinstance(v, list):
            grid_keys.append(k)
            grid_values.append(v)
        else:
            fixed[k] = v

    # グリッドがない場合も product が回るように
    if not grid_keys:
        return [fixed]

    results = []
    for values in product(*grid_values):
        d = deepcopy(fixed)
        for k, v in zip(grid_keys, values):
            d[k] = v
        results.append(d)

    return results


yaml=YAML(typ="safe")
with open("measurement/configs/grid_cfg.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f)

N_JOBS=cfg['n_jobs'] 

grid_params=cfg['params']

# measure,model,saveそれぞれで直積
measure_grid = build_section_grid(grid_params["measure"])
model_grid   = build_section_grid(grid_params["model"])
save_grid    = build_section_grid(grid_params["save"])

# BaseModelに変換
measure_grid = [MeasureConfig(**m) for m in measure_grid]
model_grid   = [RnnConfig(**m) for m in model_grid]
save_grid    = [SaveConfig(**s) for s in save_grid]

# 直積された3つでさらに直積
PARAMS_LIST = [
    {
        "measure": m,
        "model": mdl,
        "save": s,
    }
    for m, mdl, s in product(measure_grid, model_grid, save_grid)
]

print(f"全組み合わせ数: {len(PARAMS_LIST)}")
print(PARAMS_LIST[0])
