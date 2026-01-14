from dataclasses import replace
from joblib import Parallel, delayed

from common.schema import RnnConfig
from common.function import create_model,predict,save_create_data,save_predict_data
from measurement.function import *
from measurement.configs.config import RNN_CFG,SAVE_CFG,MEASURE_CFG
from measurement.configs.schema import MeasureConfig
from measurement.configs.grid_cfg import PARAMS_LIST,N_JOBS

def run_single_experiment(param):
    # パラメータ変数を用意する
    # setting.pyから、grid_params.pyで設定した部分だけを変更する形で用意する
    cource=param["COURCE"]
    measure_cfg: MeasureConfig = replace(
        MEASURE_CFG,
        train_cources=cource["TRAIN"],
        val_cources=cource["VAL"],
        predict_cource=cource["PREDICT"]
    )

    rnn_cfg: RnnConfig = replace(
        RNN_CFG,
        rnn_class=param["RNN_TYPE"],
        optimizer_class=param["OPTIMIZER"],
        input_len=param["INPUT_LEN"],
        hidden_nums=param["HIDDEN_NUMS"],
        out_steps_num=param["OUT_STEPS_NUM"],
        batch_size=param["BATCH_SIZE"],
        learning_rate=param["LEARNING_RATE"],
    )

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
        SAVE_CFG,
    )

    model = create_result["model"]
    
    csv_path= f"./measurement/result/WAVE{MEASURE_CFG.predict_cource:04d}/result_n{"t" if MEASURE_CFG.data_axis=="time" else "d"}-001.csv" 
    data_csv = pd.read_csv(csv_path, usecols=["ReceivedPower[dBm]"])
    measure_data = data_csv.values.astype(np.float64) # csv用のデータ構造からnumpy配列に変換

    result=predict(
        model,
        measure_data,
        scaler,
        RNN_CFG.input_len,
        SAVE_CFG.plot_start,
        SAVE_CFG.plot_range,
    )
    
    save_predict_data(
        run_id,
        result["true_data"],
        result["predict_data"],
        result["rmse"],
        result["rmse"],
        result["predict_result_figure"],
        SAVE_CFG,
    )

    return run_id

if __name__ == "__main__":
    print(f"\n\n{len(PARAMS_LIST)}のパラメータの組み合わせを実行します\n常に{N_JOBS}個の処理を並列実行します")
    Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_single_experiment)(params) for params in PARAMS_LIST
    )
