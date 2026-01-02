from dataclasses import replace

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

for params in PARAMS_LIST:
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
    )

    print("\n\n")
    print("######モデル作成の実行結果######")

    run_id = save_create_data(
        result["model"],
        result["history_figure"],
        result["training_time"],
        SAVE_CFG,
        fading_cfg,
        rnn_cfg,
    )

    print(f"実行時間:{result['training_time']:.2f}秒")
    print("##############################")

    fading_data = calc_nakagami_rice_fading(fading_cfg)

    model = result["model"]
    result = predict(
        model, fading_data, rnn_cfg.input_len, SAVE_CFG.plot_start, SAVE_CFG.plot_range
    )

    print("\n\n")
    print("########予測の実行結果########")

    save_predict_data(
        run_id,
        result["true_data"],
        result["predict_data"],
        result["rmse"],
        result["predict_result_figure"],
        SAVE_CFG,
    )

    print(f"rmse:{result['rmse']:.2f}")
    print("##############################")
