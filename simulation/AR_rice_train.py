#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rayleigh フェージング学習
  - 学習波: simu_wave_0001〜, 予測波: simu_wave_0000
"""
###FIXME 生成データで予測用と学習用のデータに区別をつける フォルダを分けるなど
import numpy as np
from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from ruamel.yaml import YAML

from common.utils.AR_model import fit_ar_model
from common.utils.func import second_to_hms
from simulation.configs.fading_schema import RiceConfig
from function import make_rice_fading

#NOTE 反射波のみで実験を行う場合、target_k_db=0にする
# lは直接波数+反射波数
RICE_CFG = RiceConfig(
    data_num=4000, c=3.0e8, f=1.298e9, delta_d=0.005, l=30, target_k_db=5.0
)
SEED = 42  # 乱数初期シード
RND = RandomState(SEED)

# パラメータ設定
NUM_SIMU_WAVES = 16  # 学習試行数 (予測コースの続きから～最後まで)
NUM_PRED_WAVES = 3  # 予測試行数 (0000〜00xx)
AR_ORDERS = [10, 50]  # 入力データ数

# 出力ディレクトリ
OUT_DIR = Path("result_kama") / "rayleigh"
SIMU_DIR = OUT_DIR / "simu_waves"
SIMU_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.perf_counter()

# データ読み込み
nlos_waves_db = []
# TODO:予測と学習で分ける
for i in tqdm(range(NUM_PRED_WAVES + NUM_SIMU_WAVES), desc="Simulating"):
    nlos_h = make_rice_fading(RICE_CFG, RND)
    nlos_db = 10 * np.log10(np.clip(np.abs(nlos_h) ** 2, 1e-12, None))
    distance_data=np.arange(RICE_CFG.data_num) * RICE_CFG.delta_d # サンプリング時の距離のデータ

    pd.DataFrame({
        "x[m]":distance_data,
        "Pr_db": nlos_db
    }).to_csv(
        SIMU_DIR / f"simu_wave_{i:04d}.csv", index=False
    )

    # レイリー波形の画像を保存
    plt.figure(figsize=(8, 3))
    plt.plot(nlos_db, lw=0.8)
    plt.title(f"Rayleigh Simu {i:04d}")
    plt.xlabel("Sample")
    plt.ylabel("Pr [dB]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SIMU_DIR / f"simu_wave_{i:04d}.png")
    plt.close()
    nlos_waves_db.append(nlos_db)
nlos_waves_db = np.array(nlos_waves_db)

# 予測用と学習用にデータを分離
predict_waves_db = nlos_waves_db[:NUM_PRED_WAVES]
train_waves_db = nlos_waves_db[NUM_PRED_WAVES:]

# 正規化用にデータを加工 
train_waves_flatten = np.concatenate(train_waves_db)
train_waves_vector_column=train_waves_flatten.reshape(-1, 1) #(データ,特徴量の数=1)の形にする

# 正規化
scaler = StandardScaler()
scaler.fit(train_waves_vector_column)
train_waves_scaled = scaler.transform(train_waves_vector_column).reshape(
    train_waves_db.shape
)
joblib.dump(scaler, OUT_DIR / "StandardScaler.pkl")

# AR学習
coefs_dict = {}
for order in AR_ORDERS:
    coefs = fit_ar_model(train_waves_flatten, order)
    coefs_dict[f"order-{order}"]=coefs.tolist()
yaml=YAML()
with open(OUT_DIR/"ar_coefs.yaml", "w") as f:
    yaml.dump(coefs_dict, f)

# パラメータ保存
params = {
    "dd": RICE_CFG.delta_d,
    "ar_orders": AR_ORDERS,
    "num_pred_waves": NUM_PRED_WAVES,
}
with (OUT_DIR/"params.yaml").open("w") as f:
    yaml.dump(params, f)

# 時間保存
elapsed = time.perf_counter() - start_time
h,m,s=second_to_hms(elapsed)

with (OUT_DIR/"metrics.yaml").open("w") as f:
    yaml.dump({
        "train_time_sec":elapsed,
        "train_time_hms":f"{h:02d}:{m:02d}:{s:06.3f}"
    },f)
        
