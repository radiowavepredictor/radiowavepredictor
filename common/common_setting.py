import os
from datetime import datetime,timezone,timedelta

### 予測結果のグラフの表示範囲 ###
PLOT_START = 0
PLOT_RANGE = 200 #グラフとして表示する範囲

### mlflow(実験データ管理ツール)における設定 ###
EXPERIMENT_NAME="simu-data-3000-set-15"
JST = timezone(timedelta(hours=9))
RUN_NAME = datetime.now(JST).strftime("%Y_%m_%d_%H_%M")
USE_MLFLOW=True #mlflowを使うかどうか(mlflowが使えない環境ではFalseを指定)

### json保存時の設定 ###
BASE_DIR = "exp_runs"
SAVE_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME, RUN_NAME)

