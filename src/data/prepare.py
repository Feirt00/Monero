import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, WINDOW_SIZE, TRAIN_SPLIT

def preparar_datos():
    """
    Lee el CSV bruto de Monero, realiza la limpieza, escala los datos,
    crea las ventanas deslizantes de tiempo y las exporta a un archivo .npz
    incluyendo metadatos, fechas y valores reales para evaluación.
    """
    # 1. Definir rutas de entrada y salida
    archivo_csv = os.path.join(DATA_RAW_DIR, "xmr_usd_diario.csv")
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    archivo_npz = os.path.join(DATA_PROCESSED_DIR, "dataset.npz")
    
    if not os.path.exists(archivo_csv):
        raise FileNotFoundError(f"No se encontró el archivo CSV en {archivo_csv}. Ejecuta primero download.py")

    print("Leyendo datos crudos desde el CSV...")
    df = pd.read_csv(archivo_csv)

    # 2. Limpieza básica de datos
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    
    # Seleccionamos la columna de precios de cierre
    datos = df[["Close"]].dropna().values.astype("float32")
    
    # Guardamos también las fechas asociadas si existen
    fechas = df["Date"].values if "Date" in df.columns else None

    print(f"Total de registros limpios a procesar: {len(datos)}")

    # 3. División estricta de Train/Test
    split_idx = int(len(datos) * TRAIN_SPLIT)
    train_data = datos[:split_idx]
    test_data = datos[split_idx:]
    
    # Fechas correspondientes al conjunto de prueba (ajustadas por la ventana deslizante)
    # Más adelante se alinean con las ventanas de test

    # 4. Escalado de datos (Ajustando el scaler ÚNICAMENTE con el conjunto de entrenamiento)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Unimos todo escalado para armar las ventanas de manera continua sin romper secuencias
    datos_escalados = np.vstack((train_scaled, test_scaled))

    # 5. Función para crear ventanas de tiempo (Sliding Window)
    def crear_ventanas(dataset, window_size):
        X, y = [], []
        for i in range(window_size, len(dataset)):
            X.append(dataset[i - window_size:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    X, y = crear_ventanas(datos_escalados, WINDOW_SIZE)

    # Reajustamos X para que tenga la forma tridimensional que exigen las redes LSTM: [muestras, tiempo, caracteristicas]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Volvemos a dividir X e y respetando el corte de entrenamiento y prueba
    train_size_adjusted = len(train_scaled) - WINDOW_SIZE
    
    X_train = X[:train_size_adjusted]
    y_train = y[:train_size_adjusted]
    X_test = X[train_size_adjusted:]
    y_test = y[train_size_adjusted:]

    # Extraer los valores reales (en USD) y las fechas exactas para el conjunto de test ajustado
    y_test_real = test_data[WINDOW_SIZE:] if len(test_data) > WINDOW_SIZE else test_data
    test_dates = fechas[split_idx + WINDOW_SIZE:] if fechas is not None else None

    # 6. Guardar todo empaquetado en formato .npz cumpliendo el contrato completo
    np.savez(
        archivo_npz,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_test_real=y_test_real,
        test_dates=test_dates,
        scaler=scaler,                  # Scaler completo guardado de forma segura
        scaler_min=scaler.data_min_,
        scaler_max=scaler.data_max_
    )

    print(f"¡Datos preparados y guardados con éxito en: {archivo_npz}")
    print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma de X_test:  {X_test.shape},  y_test:  {y_test.shape}")


if __name__ == "__main__":
    preparar_datos()