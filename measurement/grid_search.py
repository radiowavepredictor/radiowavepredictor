from joblib import Parallel, delayed

from common import RnnConfig,SaveConfig
from common import create_model,predict
from common import ExperimentsSaver

from function import *
from configs.schema import MeasureConfig
from configs.grid_cfg import PARAMS_LIST,N_JOBS

def run_single_experiment(param):
    # パラメータ変数を用意する
    # setting.pyから、grid_params.pyで設定した部分だけを変更する形で用意する
    measure_cfg: MeasureConfig = param["measure"]
    rnn_cfg: RnnConfig = param["model"]
    save_cfg:SaveConfig = param["save"]

    dataset, val_dataset,scaler = make_learning_dataset(measure_cfg, rnn_cfg)

    create_result = create_model(
        dataset,
        val_dataset,
        rnn_cfg,
        verbose=0,
    )

    model = create_result["model"]
    
    csv_path= Path("result")/f"WAVE{measure_cfg.cource.predict:04d}"/f"result_n{'t' if measure_cfg.data_axis=='time' else 'd'}-001.csv" 
    data_csv = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
    measure_data = data_csv.values.astype(np.float64) # csv用のデータ構造からnumpy配列に変換

    result=predict(
        model,
        measure_data,
        scaler,
        rnn_cfg,
        save_cfg.plot_start,
        save_cfg.plot_range,
        measure_cfg.sampling_rate
    )
    params={
        **measure_cfg.model_dump(), #辞書型に変換
        **rnn_cfg.model_dump()
    }
    metrics={
        "train_time":create_result["training_time"],
        "predict_time":result.predict_time,
        **result.rmse
    }
    figures={
        "history":create_result["history_figure"],
        "predict_figure":result.predict_figure
    }
    save=ExperimentsSaver(
        model=model,
        params=params,
        metrics=metrics,
        figures=figures,
        npys={"true_data":result.true_data,"predict_data":result.predict_data},
        pkls={"scaler":scaler}
    )
    save.save(save_cfg)

if __name__ == "__main__":
    print(f"\n\n{len(PARAMS_LIST)}のパラメータの組み合わせを実行します\n常に{N_JOBS}個の処理を並列実行します")
    Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_single_experiment)(params) for params in PARAMS_LIST
    )
