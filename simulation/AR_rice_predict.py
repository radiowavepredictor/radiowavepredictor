#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import time
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.preprocessing import StandardScaler
import joblib

from common.utils.AR_model import ar_predict_no_bias
from common.utils.func import second_to_hms,calc_rmse

PREDICT_POINTS:list[int] = [1, 3, 5]   # 予測点のリスト（1 = 1点先予測）
PLOT_RANGE = np.arange(0,175)        # 横表示範囲 [min, max) min~max番目のデータをplotする

RESULT_DIR = Path("result_kama")/"rayleigh" 

yaml=YAML()
#train時のデータ読み込み
with (RESULT_DIR/"params.yaml").open("r") as f:
    TRAIN_PARAMS = yaml.load(f)
    
with (RESULT_DIR/"ar_coefs.yaml").open("r") as f:
    COEFS_DICT = yaml.load(f)
    
SCALER:StandardScaler = joblib.load(RESULT_DIR/"StandardScaler.pkl")

# 時間計測開始
start_time = time.perf_counter()

# 出力先 
#TODO 名前を predictしたdataを保存する場所って分かるようにする
WAVES_ROOT_DIR = RESULT_DIR/ "simu_waves_predict"
WAVES_ROOT_DIR.mkdir(parents=True, exist_ok=True)

rmse_all =[] 

#FIXME for文3重で見づらい どこかで関数で抜き出したほうがいい
for pred_point in PREDICT_POINTS:
    for pred_wave_idx in range(TRAIN_PARAMS["num_pred_waves"]):
        # 予測に使うデータを読み込み
        # TODO:変数名変えたほうがいいかも
        pr_df = pd.read_csv(RESULT_DIR/"simu_waves"/f"simu_wave_{pred_wave_idx:04d}.csv")
        
        pr_db = pr_df["Pr_db"].to_numpy()
        x=pr_df["x[m]"].to_numpy()
        # TODO 入力特徴量が複数の場合にも対応させる
        pr_norm = SCALER.transform(pr_db.reshape(-1,1)).reshape(pr_db.shape)        

        result_df = pd.DataFrame({
            "Index": np.arange(len(pr_db)),
            "x[m]":x,
            "Actual": pr_db
        })
        
        # 予測
        for order in TRAIN_PARAMS["ar_orders"]:
            coefs = COEFS_DICT[f"order-{order}"]
            pred_norm = ar_predict_no_bias(pr_norm, coefs, pred_point)
            #NOTE 予測データと予測元データをcsvに入れるため、長さをそろえている
            # (入力の長さ+予測点)だけ予測データはデータが少ないので、nanで埋める
            pred_norm_full = np.concatenate([np.full(order + pred_point - 1, np.nan), pred_norm]) 
            #TODO 入力特徴量が複数の場合にも対応させる
            pred_db = SCALER.inverse_transform(pred_norm_full.reshape(-1,1)).reshape(pred_norm_full.shape)

            result_df[f"order-{order}"] = pred_db
            rmse_db = calc_rmse(pr_db, pred_db)
            
            rmse_all.append({"pred_wave": pred_wave_idx, "order": order, "pred_point": pred_point,"RMSE_dB": rmse_db})

        # 予測結果保存
        # TODO ここも変数名変えたい 予測結果であることを明示したい
        wave_dir = WAVES_ROOT_DIR / f"wave_{pred_wave_idx:04d}"
        wave_dir.mkdir(parents=True, exist_ok=True)
 
        result_df.to_csv(wave_dir/ f"point-{pred_point}.csv",index=False)

        # plot
        plt.rcParams["font.size"] = 17
        
        fig = plt.figure(figsize=(12, 5.25))
        plt.xlabel("移動距離[m]")
        plt.ylabel("受信電力レベル[dB]")

        # 予測元データのplot
        #FIXME plotは関数に抜き出すか別のコードにするかして分けたい
        plt.plot(
            x[PLOT_RANGE],
            pr_db[PLOT_RANGE],
            color="black",
            alpha=0.5,
            linewidth=2.25,
            marker="o",           # marker（丸）
            markersize=5,         
            markerfacecolor="black",
            markeredgecolor="black",
            label="予測元値", #TODO これださい
        )

        for order in TRAIN_PARAMS["ar_orders"]:
            plt.plot(
                x[PLOT_RANGE],
                result_df[f"order-{order}"][PLOT_RANGE],
                linestyle="--",
                alpha=0.9,
                linewidth=2.25,
                marker="o",           # marker（丸）
                markersize=5,         
                label=f"予測値(AR-{order})",
            )

        plt.legend()
        plt.grid()

        fig.savefig(wave_dir / f"point-{pred_point}.svg", bbox_inches="tight")
        plt.close(fig)
        
pd.DataFrame(rmse_all).to_csv(RESULT_DIR / "rmse.csv", index=False) #TODO できればmetricsに一緒に保存したいけど、そのままだと見づらい

# 時間計測終了
elapsed = time.perf_counter() - start_time
h,m,s=second_to_hms(elapsed)

# metics保存
with (RESULT_DIR/"metrics.yaml").open("r") as f:
    metrics = yaml.load(f)

metrics["predict_time_sec"]=float(elapsed) #TODO これfloatって必要なの?
metrics["predict_time_hms"]=f"{h:02d}:{m:02d}:{s:06.3f}"

with open(RESULT_DIR/"metrics.yaml", "w") as f:
    yaml.dump(metrics, f)

print(f"⏱ Finish timer (predict): {h:02d}:{m:02d}:{s:06.3f} (hh:mm:ss.sss)")