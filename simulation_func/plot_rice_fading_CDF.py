###### 電力と累積確率関数のグラフを表示する #######
###### データが正しく生成されているの確認用 #######
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from func import *

k_rice_list = [-np.inf, 0, 3, 5, 10, 20, 40]

plt.figure(figsize=(9,6))

for k_rice in k_rice_list:
    fading = calc_nakagami_rice_fading(k_rice)
    power_linear = np.abs(fading)**2

    power_db = 10 * np.log10(power_linear / np.mean(power_linear)) #正規化とdb変換

    # CDF 計算
    sorted_power = np.sort(power_db)
    cdf = np.arange(1, len(sorted_power)+1) / len(sorted_power) * 100

    # ラベル表示
    if np.isinf(k_rice):
        label = "K = -∞ (Rayleigh)"
    else:
        label = f"K = {k_rice}"

    plt.plot(sorted_power, cdf, label=label)

plt.yscale("log")           # 対数スケール
plt.xlim(-40,10)
plt.ylim(0.1, 100)          # y軸 0.1% 〜 100%

# 指数表記(10^-1,10^0,10^1,10^2)になっているところを通常表記（0.1, 1, 10, 100）にする
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())

plt.xlabel("Relative Received Power (dB)")
plt.ylabel("Cumulative Probability (%)")
plt.title("Nakagami–Rice Fading CDF for Various K Factors")
plt.grid(True)
plt.legend()
plt.show()

