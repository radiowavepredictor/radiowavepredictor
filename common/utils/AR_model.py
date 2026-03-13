import numpy as np

def fit_ar_model(time_series, order):
    """
    単変量時系列に対してAR(order)モデルを最小二乗法で推定する関数

    Parameters
    ----------
    time_series : 1次元配列（numpy arrayなど）
        時系列データ
    order : int
        ARモデルの次数（過去何ステップを使うか）

    Returns
    -------
    coefs : numpy array
        ARモデルの係数 [a1, a2, ..., a_order]
    """
    # 説明変数 X を作成
    X = np.array([
        time_series[i:i + order][::-1]  # 過去データを逆順にして最新を左に
        for i in range(len(time_series) - order)
    ])

    # 目的変数 y
    y = time_series[order:]

    # 正規方程式で最小二乗解
    XtX = X.T @ X + 1e-6 * np.eye(order)  # 数値安定化
    coefs = np.linalg.pinv(XtX) @ (X.T @ y)

    return coefs

def ar_predict_no_bias(series, coeffs, predict_points=1):
    order = len(coeffs)
    pred = []
    data = series.tolist()
    if len(series) < order:
        return np.array([])
    for i in range(len(series) - order - predict_points + 1):
        window = data[i:i + order]
        for _ in range(predict_points):
            p = np.dot(coeffs[::-1], window[-order:])
            window.append(p)
        pred.append(window[-1])
    return np.array(pred)
