import os
import json
import argparse
import numpy as np

# Importaciones directas de configuración y registro
from src.config import DATA_PROCESSED_DIR, MODELS_DIR, RESULTS_METRICS_DIR
from src import REGISTRY

def main():
    print(">>> Ejecutando script de entrenamiento (CLI)...")
    
    parser = argparse.ArgumentParser(description="Script CLI para entrenamiento y evaluación de modelos.")
    parser.add_argument("--modelo", type=str, required=True, choices=list(REGISTRY.keys()), help="Nombre del modelo a entrenar.")
    parser.add_argument("--objetivo", type=str, default="precio", choices=["precio", "retorno"], help="Variable objetivo.")
    parser.add_argument("--semilla", type=int, default=0, help="Semilla aleatoria para reproducibilidad.")
    
    args = parser.parse_args()

    print(f"--- Entrenando modelo: {args.modelo.upper()} | Objetivo: {args.objetivo} | Semilla: {args.semilla} ---")

    # 1. Cargar el dataset procesado
    dataset_path = os.path.join(DATA_PROCESSED_DIR, "dataset.npz")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No se encontró {dataset_path}. Ejecuta primero prepare.py")

    data = np.load(dataset_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_test_real = data["y_test_real"] if "y_test_real" in data else None
    scaler = data["scaler"].item() if "scaler" in data else None

    # Configurar semilla
    np.random.seed(args.semilla)
    import tensorflow as tf
    tf.random.set_seed(args.semilla)

    # 2. Instanciar y entrenar el modelo
    model_class = REGISTRY[args.modelo]
    model = model_class()

    print("Iniciando fase de entrenamiento (fit)...")
    model.fit(X_train, y_train)

    # 3. Generar predicciones en test
    print("Generando predicciones sobre el conjunto de test...")
    preds = model.predict(X_test, scaler=scaler)

    if y_test_real is not None:
        y_real_eval = y_test_real.flatten()
    elif scaler is not None:
        y_real_eval = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    else:
        y_real_eval = y_test.flatten()

    min_len = min(len(y_real_eval), len(preds))
    y_real_eval = y_real_eval[:min_len]
    preds = preds[:min_len]

    # 4. Calcular métricas
    from src.evaluation.metrics import mae, rmse, mape, acierto_direccion
    score_mae = mae(y_real_eval, preds)
    score_rmse = rmse(y_real_eval, preds)
    score_mape = mape(y_real_eval, preds)
    score_dir = acierto_direccion(y_real_eval, preds)

    metrics_dict = {
        "modelo": args.modelo,
        "objetivo": args.objetivo,
        "semilla": args.semilla,
        "mae": float(score_mae),
        "rmse": float(score_rmse),
        "mape": float(score_mape),
        "acierto_direccion": float(score_dir)
    }

    print(f"\nResultados de Evaluación:")
    for k, v in metrics_dict.items():
        print(f"  - {k}: {v}")

    # 5. Guardar artefactos y métricas en JSON
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)

    if hasattr(model, "save"):
        model_filename = os.path.join(MODELS_DIR, f"{args.modelo}_{args.objetivo}_seed{args.semilla}.keras")
        try:
            model.save(model_filename)
            print(f"Modelo guardado en: {model_filename}")
        except Exception as e:
            print(f"Aviso al guardar modelo: {e}")

    metrics_filename = os.path.join(RESULTS_METRICS_DIR, f"{args.modelo}_{args.objetivo}_seed{args.semilla}.json")
    with open(metrics_filename, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"¡Métricas guardadas con éxito en: {metrics_filename}!\n")

if __name__ == "__main__":
    main()