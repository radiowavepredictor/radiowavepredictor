####レイリーフェージングのみ(反射波のみ)でのデータ生成を行いプロットする####
#### データが正しく生成されているかどうかの確認コード #####
import numpy as np
import matplotlib.pyplot as plt

from function import make_nlos_fading
from configs.config import SIMULATION_CFG

rnd=np.random.RandomState(0)
nlos_fading=make_nlos_fading(SIMULATION_CFG,rnd)

power = np.abs(nlos_fading)**2

power_db = 10 * np.log10(power)

# ---- グラフ表示 ----
plt.figure(figsize=(10, 4))
plt.xlim(0,5)
plt.ylim(-30,10)
x = np.arange(SIMULATION_CFG.data_num) * SIMULATION_CFG.delta_d
plt.plot(x, power_db, label="Rayleigh fading power")
plt.xlabel("Distance moved [m]")
plt.ylabel("Relative Signal Power [dB]")
plt.title("Rayleigh Fading Power vs Distance (Linear Motion)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
