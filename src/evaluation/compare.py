import os
import glob
import json
import pandas as pd
from src.config import RESULTS_METRICS_DIR

def comparar_modelos():
    """
    Lee todos los archivos JSON de métricas en results/metrics/,
    los consolida en un DataFrame de Pandas, los ordena por MAE
    y exporta una tabla comparativa en formato CSV.
    """
    patron_json = os.path.join(RESULTS_METRICS_DIR, "*.json")
    archivos = glob.glob(patron_json)
    
    if not archivos:
        print(f"No se encontraron archivos de métricas en {RESULTS_METRICS_DIR}. Entrena algún modelo primero.")
        return
    
    registros = []
    for archivo in archivos:
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                data = json.load(f)
                registros.append(data)
        except Exception as e:
            print(f"Error leyendo el archivo {archivo}: {e}")
            
    if not registros:
        print("No hay registros válidos para comparar.")
        return

    # Crear DataFrame y ordenar de menor a mayor MAE (mejor precisión)
    df = pd.DataFrame(registros)
    if "mae" in df.columns:
        df = df.sort_values(by="mae", ascending=True).reset_index(drop=True)

    # Guardar en CSV dentro de la misma carpeta de métricas
    output_csv = os.path.join(RESULTS_METRICS_DIR, "comparacion.csv")
    df.to_csv(output_csv, index=False)
    
    print(f"¡Tabla comparativa generada con éxito en: {output_csv}\n")
    print("=" * 70)
    print("TABLA COMPARATIVA DE MODELOS (Ordenados por MAE):")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    comparar_modelos()