#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# ============================================================
# Nakagami-Rice フェージング学習
#   - 学習波: simu_wave_{num_pred_waves:04d}〜, 予測波: simu_wave_0000〜
# ============================================================

# パラメータ設定（ユーザー設定）
NUM_SIMU_WAVES = 1  # 学習コース数
NUM_PRED_WAVES = 0   # 予測コース数（例: 3なら0000〜0002）
NUM_SAMPLES = 100
PREDICT_TYPE = "d"  # "t"=time, "d"=distance
DD = 0.03925  # distance step [m]
F_C = 1.298e9
L = 30
SEED = 42
PT, LOSS = 0.0, 0.0
AR_ORDERS = [10, 50]

# r_i & r0 設定
"""
r_iは手動で、r_0はライスファクタKから自動 or 手動で入力
"""
R_I_MODE = "constant"   # "constant" or "rayleigh"
R_I_CONST = 1.0         # NLoS の強さ（固定したい値）
R_I_SCALE = 2.2         # NLoS の強さ（固定したい値）
K_TARGET_DB = 4.0       # ← ここに“目標K[dB]”を指定して r0 を決める
R0 = 2.0                # K_target_dB=None のとき r0 を手動指定

if K_TARGET_DB:
    K_lin = 10 ** (K_TARGET_DB / 10.0)
    if R_I_MODE== "constant":
        # P_NLoS ≈ r_i_const^2 とみなして r0 を計算
        R0 = R_I_CONST* np.sqrt(K_lin)
        print(f"[K指定] K_target = {K_TARGET_DB:.2f} dB, "
              f"r_i_const = {R_I_CONST:.4f} → r0 = {R0:.4f}")
    elif R_I_MODE== "rayleigh":
        # Rayleigh(σ) の平均パワーは 2σ^2 → P_NLoS ≈ 2 * r_i_scale^2
        r0 = np.sqrt(2.0) * R_I_SCALE* np.sqrt(K_lin)
        print(f"[K指定] K_target = {K_TARGET_DB:.2f} dB, "
              f"r_i_scale = {R_I_SCALE:.4f} → r0 = {r0:.4f}")
    else:
        raise ValueError(f"Invalid r_i_mode: {R_I_MODE}")

# ---- 出力フォルダ ----
out_dir = "./result/rice"
os.makedirs(out_dir, exist_ok=True)
simu_dir = os.path.join(out_dir, "simu_waves")
os.makedirs(simu_dir, exist_ok=True)
k_result_txt = os.path.join(out_dir, "k_factor_result.txt")

# ============================================================
# 関数定義
# ============================================================

def generate_rice_wave(num_samples, f_c, L, dd_step, r0, seed=0):
    rnd = np.random.RandomState(seed)
    c = 3.0e8
    lam = c / f_c
    Lm = L - 1

    if R_I_MODE == "constant":
        r_i = np.full(Lm, R_I_CONST)
    elif R_I_MODE == "rayleigh":
        r_i = rnd.rayleigh(scale=R_I_SCALE, size=Lm)
    else:
        raise ValueError(f"Invalid r_i_mode: {R_I_MODE}")

    # 各パスの固定フェーズ・到来角
    theta = rnd.rand(Lm) * 2 * np.pi
    phi = rnd.rand(Lm) * 2 * np.pi

    # 初期距離差
    dd = np.zeros(Lm)
    dd_los = 0.0

    h = np.zeros(num_samples, dtype=complex)
    h_los = np.zeros(num_samples, dtype=complex)
    h_nlos = np.zeros(num_samples, dtype=complex)

    for t in range(num_samples):
        dd += dd_step
        dd_los += dd_step

        term = r_i * np.exp(1j * theta) * np.exp(
            1j * (2 * np.pi / lam) * dd * np.cos(phi)
        )
        h_nlos[t] = (1.0 / np.sqrt(Lm)) * np.sum(term)
        h_los[t] = r0 * np.exp(1j * (2 * np.pi / lam) * dd_los)
        h[t] = h_los[t] + h_nlos[t]

    Pr_db = 10 * np.log10(np.clip(np.abs(h) ** 2, 1e-12, None)) + PT - LOSS
    return Pr_db, h_los, h_nlos


def estimate_K_power(h_los, h_nlos):
    num = np.mean(np.abs(h_los) ** 2)
    den = max(np.mean(np.abs(h_nlos) ** 2), 1e-15)
    K_lin = num / den
    return float(K_lin), float(10 * np.log10(K_lin))

# ============================================================
# メイン処理
# ============================================================
# 時間計測開始
t0 = time.perf_counter()
print("⏱ Start timer")

print("Generating Rice simulated waves...")
total_waves = NUM_SIMU_WAVES + NUM_PRED_WAVES  # 例: num_simu_waves=17, num_pred_waves=3 → 0000〜0019

train_series_list = []
per_wave_stats = []
K_lin_list, K_dB_list = [], []

for i in range(total_waves):
    #los 見通し内 nlosが見通し外
    sim_db, h_los, h_nlos = generate_rice_wave(NUM_SAMPLES, F_C, L, DD,R0, SEED+ i)
    print(h_nlos[:5])
    '''
    K_lin, K_dB = estimate_K_power(h_los, h_nlos)
    K_lin_list.append(K_lin)
    K_dB_list.append(K_dB)

    # 保存
    fname = f"simu_wave_{i:04d}.csv"
    pd.DataFrame({"Pr_db": sim_db}).to_csv(os.path.join(simu_dir, fname), index=False)

    plt.figure(figsize=(8, 3))
    plt.plot(sim_db, lw=0.8)
    plt.title(f"Rice Simu {i:04d}")
    plt.xlabel("Sample")
    plt.ylabel("Pr [dB]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(simu_dir, f"simu_wave_{i:04d}.png"))
    plt.close()

    # 各波のmean/std
    mean_i = float(np.mean(sim_db))
    std_i = float(np.std(sim_db))
    if std_i == 0: std_i = 1.0
    per_wave_stats.append({"wave": f"simu_wave_{i:04d}", "mean": mean_i, "std": std_i})

    # 学習波のみ収集
    if i >= NUM_PRED_WAVES:
        train_series_list.extend(sim_db.tolist())
    '''
'''
# ---- グローバルmean/std ----
global_mean = float(np.mean(train_series_list))
global_std = float(np.std(train_series_list))
if global_std == 0: global_std = 1.0
print(f"Global mean = {global_mean:.3f}, std = {global_std:.3f}")

# ---- norm_params.csv ----
norm_rows = [{"wave": "train", "mean": global_mean, "std": global_std}]
norm_rows.extend(per_wave_stats)

pd.DataFrame(norm_rows, columns=["wave", "mean", "std"]).to_csv(
    os.path.join(out_dir, "norm_params.csv"), index=False
)
print("✅ Saved normalization parameters (global + per-wave stats)")

# ---- K-factor summary ----
with open(k_result_txt, "w") as f:
    f.write(f"r0 = {r0:.6f}\n")
    f.write(f"K_lin_mean = {np.mean(K_lin_list):.6f}\n")
    f.write(f"K_dB_mean = {np.mean(K_dB_list):.6f} dB\n")
print(f"✅ Saved K-factor summary → {k_result_txt}")


# ---- AR係数推定 ----
combined_normed = np.concatenate([
    (pd.read_csv(os.path.join(simu_dir, f"simu_wave_{i:04d}.csv"))["Pr_db"].values - global_mean) / global_std
    for i in range(num_pred_waves, total_waves)
])

coefs = []
for order in ar_orders:
    n = len(combined_normed)
    if n <= order:
        print(f"Skip AR({order}) due to short data.")
        continue
    X = np.array([combined_normed[i:i+order][::-1] for i in range(n - order)])
    y = combined_normed[order:]
    XtX = X.T @ X + 1e-6 * np.eye(order)
    a = np.linalg.pinv(XtX) @ (X.T @ y)
    for j, c in enumerate(a):
        coefs.append({"AR_order": order, "coef_index": j, "coef_value": float(c)})

pd.DataFrame(coefs).to_csv(os.path.join(out_dir, "ar_coeffs.csv"), index=False)

# ---- params.csv ----
params = {
    "r0": r0,
    "r_i_mode": r_i_mode,
    "r_i_const": r_i_const,
    "r_i_scale": r_i_scale,
    "Pt_dBm": Pt,
    "Loss_dB": Loss,
    "num_simu_waves": num_simu_waves,
    "num_pred_waves": num_pred_waves,
    "dd": float(dd),
    "dt": float(dt),
    "predict_type": predict_type,
    "ar_orders": ",".join(map(str, ar_orders)),
}
pd.DataFrame(params.items(), columns=["param", "value"]).to_csv(
    os.path.join(out_dir, "params.csv"), index=False
)

print("\n✅ Rice training finished (GLOBAL normalization mode).")


# 時間計測終了
elapsed = time.perf_counter() - t0
h = int(elapsed // 3600)
m = int((elapsed % 3600) // 60)
s = elapsed % 60

pd.DataFrame([{
    "elapsed_sec": float(elapsed),
    "elapsed_hms": f"{h:02d}:{m:02d}:{s:06.3f}"
}]).to_csv(os.path.join(out_dir, "timer (train).csv"), index=False)

print(f"⏱ Finish timer (train): {h:02d}:{m:02d}:{s:06.3f} (hh:mm:ss.sss)")
'''