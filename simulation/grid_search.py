from joblib import Parallel, delayed

from common.function.model import create_model
from common.function.save import save_create_data
from common.schema.config import RnnConfig,SaveConfig
from simulation.function import *
from simulation.configs.schema import SimulationConfig
from simulation.configs.grid_cfg import PARAMS_LIST,N_JOBS

def run_single_experiment(param):
    # パラメータ変数を用意する
    # setting.pyから、grid_params.pyで設定した部分だけを変更する形で用意する
    simulation_cfg: SimulationConfig = param["simulation"]
    rnn_cfg: RnnConfig = param["model"]
    save_cfg:SaveConfig=param["save"]

    (dataset, val_dataset), scaler= load_fading_dataset(simulation_cfg, rnn_cfg)

    create_result = create_model(
        dataset,
        val_dataset,
        rnn_cfg,
        verbose=0,
    )

    run_id = save_create_data(
        create_result["model"],
        scaler,
        create_result["history_figure"],
        create_result["training_time"],
        simulation_cfg,
        rnn_cfg,
        save_cfg,
    )

    model = create_result["model"]
    
    first_result,rmse_mean_arr=evaluate_model(model,scaler,simulation_cfg,rnn_cfg,save_cfg)
    
    wrap_save_predict_data(
        run_id,
        first_result["true_data"],
        first_result["predict_data"],
        first_result["predict_time"],
        first_result["rmse_arr"][rnn_cfg.out_steps_num-1],
        rmse_mean_arr[rnn_cfg.out_steps_num-1],
        first_result["predict_result_figure"],
        save_cfg
    )

    return run_id

if __name__ == "__main__":
    print(f"\n\n{len(PARAMS_LIST)}のパラメータの組み合わせを実行します\n常に{N_JOBS}個の処理を並列実行します")
    Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_single_experiment)(param) for param in PARAMS_LIST
    )
