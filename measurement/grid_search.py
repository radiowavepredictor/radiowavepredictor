from joblib import Parallel, delayed

from common.schema import RnnConfig,SaveConfig
from common.function.model import create_model,predict
from common.function.save import save_create_data,save_predict_data
from measurement.function import *
from measurement.configs.schema import MeasureConfig
from measurement.configs.grid_cfg import PARAMS_LIST,N_JOBS

def run_single_experiment(param):
    # パラメータ変数を用意する
    # setting.pyから、grid_params.pyで設定した部分だけを変更する形で用意する
    measure_cfg: MeasureConfig = param["measure"]
    rnn_cfg: RnnConfig = param["model"]
    save_cfg:SaveConfig = param["save"]

    (dataset, val_dataset),scaler = load_learning_dataset(measure_cfg, rnn_cfg)

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
        measure_cfg,
        rnn_cfg,
        save_cfg,
    )

    model = create_result["model"]
    
    csv_path= Path(".")/"measurement"/"result"/f"WAVE{measure_cfg.cource.predict:04d}"/f"result_n{'t' if measure_cfg.data_axis=='time' else 'd'}-001.csv" 
    data_csv = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
    measure_data = data_csv.values.astype(np.float64) # csv用のデータ構造からnumpy配列に変換

    result=predict(
        model,
        measure_data,
        scaler,
        rnn_cfg,
        save_cfg.plot_start,
        save_cfg.plot_range,
    )
    
    save_predict_data(
        run_id,
        result["true_data"],
        result["predict_data"],
        result["rmse_arr"][rnn_cfg.out_steps_num-1],
        result["predict_result_figure"],
        save_cfg,
    )

    return run_id

if __name__ == "__main__":
    print(f"\n\n{len(PARAMS_LIST)}のパラメータの組み合わせを実行します\n常に{N_JOBS}個の処理を並列実行します")
    Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_single_experiment)(params) for params in PARAMS_LIST
    )
