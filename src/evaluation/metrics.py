import numpy as np

def mae(y_true, y_pred):
    """Calcula el Error Absoluto Medio (MAE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    """Calcula la Raíz del Error Cuadrático Medio (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    """Calcula el Error Porcentual Absoluto Medio (MAPE) en porcentaje."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Evitar división por cero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def acierto_direccion(y_true, y_pred):
    """
    Calcula el porcentaje de acierto en la dirección del cambio (subida/bajada).
    Compara el signo del cambio real respecto al anterior con el predicho.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) < 2:
        return 0.0
    
    # Dirección real y predicha basada en el cambio respecto al valor real anterior
    dir_real = np.sign(y_true[1:] - y_true[:-1])
    dir_pred = np.sign(y_pred[1:] - y_true[:-1])
    
    # Evitar divisiones o NaNs si hay ceros en np.sign
    aciertos = (dir_real == dir_pred)
    return float(np.mean(aciertos) * 100)