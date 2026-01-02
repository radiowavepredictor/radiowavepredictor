from dataclasses import replace
from joblib import Parallel, delayed

from simulation.simu_func import (
    load_fading_data,
    save_create_data,
    calc_nakagami_rice_fading,
    save_predict_data,
)
from simulation.schema_setting import FadingConfig, RnnConfig, SaveConfig
from simulation.setting import FADING_CFG, RNN_CFG, SAVE_CFG
from simulation.grid_params import PARAMS_LIST
from common.common_func import create_model, predict
from common.common_setting import BASE_DIR
import uuid


def run_single_experiment(params):
    # パラメータ変数を用意する
    # setting.pyから、grid_params.pyで設定した部分だけを変更する形で用意する
    fading_cfg: FadingConfig = replace(
        FADING_CFG,
        data_num=params["DATA_NUM"],
        data_set_num=params["DATA_SET_NUM"],
        delta_d=params["DELTA_D"],
        k_rice=params["K_RICE"],
    )

    rnn_cfg: RnnConfig = replace(
        RNN_CFG,
        rnn_class=params["RNN_TYPE"],
        optimizer_class=params["OPTIMIZER"],
        input_len=params["INPUT_LEN"],
        hidden_nums=params["HIDDEN_NUMS"],
        out_steps_num=params["OUT_STEPS_NUM"],
        batch_size=params["BATCH_SIZE"],
        learning_rate=params["LEARNING_RATE"],
    )

    run_name = uuid.uuid4().hex[:8]
    save_cfg: SaveConfig = replace(
        SAVE_CFG,
        run_name=run_name,
        save_dir=f"{BASE_DIR}/{SAVE_CFG.experiment_name}/{run_name}",
    )

    dataset, val_dataset = load_fading_data(fading_cfg, rnn_cfg)

    result = create_model(
        dataset,
        val_dataset,
        rnn_cfg.input_len,
        rnn_cfg.in_features,
        rnn_cfg.hidden_nums,
        rnn_cfg.rnn_class,
        rnn_cfg.optimizer_class,
        rnn_cfg.out_steps_num,
        rnn_cfg.learning_rate,
        rnn_cfg.epochs,
        verbose="silent",
    )

    run_id = save_create_data(
        result["model"],
        result["history_figure"],
        result["training_time"],
        save_cfg,
        fading_cfg,
        rnn_cfg,
    )

    fading_data = calc_nakagami_rice_fading(fading_cfg)

    model = result["model"]
    result = predict(
        model, fading_data, rnn_cfg.input_len, save_cfg.plot_start, save_cfg.plot_range
    )

    save_predict_data(
        run_id,
        result["true_data"],
        result["predict_data"],
        result["rmse"],
        result["predict_result_figure"],
        save_cfg,
    )

    return run_id


if __name__ == "__main__":
    print(f"{len(PARAMS_LIST)}の処理を並列実行します")
    Parallel(n_jobs=4, verbose=10)(
        delayed(run_single_experiment)(params) for params in PARAMS_LIST
    )
