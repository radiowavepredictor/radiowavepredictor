import pandas as pd
import numpy as np

def read_csv(num):
    csv_path = f"./simu_waves/simu_wave_{num:04d}.csv"
    csv_data = pd.read_csv(csv_path, usecols=["Pr_db"])
    np_data = csv_data.values.astype(np.float64)
    return np_data

csvs=[]
for i in range(203):
    csvs.append(read_csv(i))
csvs=np.array(csvs)
print(csvs[0])
