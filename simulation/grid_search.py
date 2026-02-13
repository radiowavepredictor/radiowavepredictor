from joblib import Parallel, delayed
from numpy.random import RandomState

from common import create_model
from common import ExperimentsSaver
from common import RnnConfig,SaveConfig
from function import make_rice_learning_dataset,predict_multiple_waves
from configs.schema import SimulationConfig
from configs.grid_cfg import PARAMS_LIST,N_JOBS

def run_single_experiment(param):
    # パラメータ変数を用意する
    # setting.pyから、grid_params.pyで設定した部分だけを変更する形で用意する
    simulation_cfg: SimulationConfig = param["simulation"]
    rnn_cfg: RnnConfig = param["model"]
    save_cfg:SaveConfig=param["save"]

    rnd = RandomState(0)
    dataset, val_dataset, scaler= make_rice_learning_dataset(simulation_cfg, rnn_cfg,rnd)

    create_result = create_model(
        dataset,
        val_dataset,
        rnn_cfg,
        verbose=0,
    )
    
    model = create_result["model"]
    
    first_result,rmse_mean_dict=predict_multiple_waves(model,scaler,rnd,simulation_cfg,rnn_cfg,save_cfg)
    params={**simulation_cfg.model_dump(),**rnn_cfg.model_dump()}
    metrics={
        "train_time":create_result["training_time"],
        "predict_time":first_result.predict_time,
        **first_result.rmse,
        **rmse_mean_dict
    }
    figures={
        "history":create_result['history_figure'],
        "predict_figure":first_result.predict_figure
    }
    npys={"true_data":first_result.true_data,"predict_data":first_result.predict_data}
    save=ExperimentsSaver(
        model=model,
        metrics=metrics,
        params=params,
        figures=figures,
        npys=npys,
        pkls={"scaler":scaler}
        
    )
    save.save(save_cfg)

if __name__ == "__main__":
    print(f"\n\n{len(PARAMS_LIST)}のパラメータの組み合わせを実行します\n常に{N_JOBS}個の処理を並列実行します")
    Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_single_experiment)(param) for param in PARAMS_LIST
    )
