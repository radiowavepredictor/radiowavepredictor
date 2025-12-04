####レイリーフェージングのみでのデータ生成を行いプロットする####
import numpy as np
import matplotlib.pyplot as plt
from simu_func import *

fading=calc_fading()

power = np.abs(fading)**2

power_db = 10 * np.log10(power)

# ---- グラフ表示 ----
plt.figure(figsize=(10, 4))
plt.xlim(0,5)
plt.ylim(-30,10)
x = np.arange(DATA_NUM) * DELTA_D
plt.plot(x, power_db, label="Rayleigh fading power")
plt.xlabel("Distance moved [m]")
plt.ylabel("Relative Signal Power [dB]")
plt.title("Rayleigh Fading Power vs Distance (Linear Motion)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
