import os
import sys
import json
import argparse
import datetime as dt

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_FIG_DIR,
    WINDOW_SIZE,
)
from src import REGISTRY


def _log(*args):
    # Los logs van a stderr para que stdout sea JSON valido
    print(*args, file=sys.stderr)


def _ultima_ventana(precios, scaler, ventana):
    ultimos = precios[-ventana:].reshape(-1, 1)
    escalados = scaler.transform(ultimos)
    return escalados.reshape(1, ventana, 1)


def _ultima_ventana_retornos(precios, ret_scaler, ventana):
    retornos = np.zeros_like(precios)
    retornos[1:] = np.log(precios[1:] / precios[:-1])
    ultimos = retornos[-ventana:].reshape(-1, 1)
    escalados = ret_scaler.transform(ultimos)
    return escalados.reshape(1, ventana, 1)


def _metricas_test(modelo, objetivo, semilla):
    ruta = os.path.join(RESULTS_METRICS_DIR, f"{modelo}_{objetivo}_seed{semilla}.json")
    if not os.path.exists(ruta):
        return {}
    with open(ruta, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {
        "MAE": round(m.get("mae", 0.0), 4),
        "RMSE": round(m.get("rmse", 0.0), 4),
        "MAPE": round(m.get("mape", 0.0), 4),
        "acierto_direccion": round(m.get("acierto_direccion", 0.0), 4),
    }


def _graficar_prediccion(nombre, fechas, precios, precio_predicho, cambio_pct, fecha_pred, dias, ruta):
    """Dibuja el historial reciente y el punto predicho para el dia siguiente."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(dias, len(precios))
    hist_fechas = pd.to_datetime(fechas[-n:])
    hist_precios = precios[-n:, 0]
    f_pred = pd.to_datetime(fecha_pred)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist_fechas, hist_precios, color="#1f77b4", linewidth=1.5, label="Precio real")
    ax.plot(
        [hist_fechas[-1], f_pred],
        [hist_precios[-1], precio_predicho],
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
    )
    ax.scatter([f_pred], [precio_predicho], color="#d62728", zorder=5, s=45, label="Prediccion")

    etiqueta = f"{precio_predicho:.2f} USD ({cambio_pct:+.2f}%)"
    ax.annotate(
        etiqueta,
        (f_pred, precio_predicho),
        textcoords="offset points",
        xytext=(-8, 12),
        ha="right",
        color="#d62728",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_title(f"XMR-USD - {nombre}")
    ax.set_ylabel("Precio (USD)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def predecir(modelo_nombre, objetivo, semilla, dias=90):
    ruta_csv = os.path.join(DATA_RAW_DIR, "xmr_usd_diario.csv")
    ruta_npz = os.path.join(DATA_PROCESSED_DIR, "dataset.npz")
    if not os.path.exists(ruta_csv) or not os.path.exists(ruta_npz):
        raise SystemExit(
            "Faltan datos. Ejecuta primero:\n"
            "  python -m src.data.download\n"
            "  python -m src.data.prepare"
        )

    df = pd.read_csv(ruta_csv)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)

    fechas = df["Date"].dt.strftime("%Y-%m-%d").to_numpy()
    precios = df["Close"].to_numpy(dtype="float64").reshape(-1, 1)
    precio_actual = float(precios[-1, 0])

    data = np.load(ruta_npz, allow_pickle=True)
    scaler = data["scaler"].item()
    ret_scaler = data["ret_scaler"].item()

    clase = REGISTRY[modelo_nombre]
    modelo = clase()
    nombre = f"{modelo_nombre}_{objetivo}_seed{semilla}"
    ruta_modelo = os.path.join(MODELS_DIR, nombre + modelo.extension)

    if modelo_nombre in ("lstm", "gru"):
        if not os.path.exists(ruta_modelo):
            raise SystemExit(
                f"No existe el modelo entrenado {ruta_modelo}.\n"
                f"Entrena primero: python -m src.train --modelo {modelo_nombre} "
                f"--objetivo {objetivo} --semilla {semilla}"
            )
        modelo.load(ruta_modelo)
        x = _ultima_ventana_retornos(precios, ret_scaler, WINDOW_SIZE)
        retorno_log = float(modelo.predict(x, scaler=ret_scaler)[0])
        precio_predicho = precio_actual * float(np.exp(retorno_log))

    elif modelo_nombre == "arima":
        if not os.path.exists(ruta_modelo):
            raise SystemExit(
                f"No existe el modelo entrenado {ruta_modelo}.\n"
                f"Entrena primero: python -m src.train --modelo arima --objetivo precio"
            )
        modelo.load(ruta_modelo)
        # Reajusta con toda la historia disponible para pronosticar manana
        modelo.history = list(scaler.transform(precios).flatten())
        x = _ultima_ventana(precios, scaler, WINDOW_SIZE)
        precio_predicho = float(modelo.predict(x, scaler=scaler)[0])

    else:
        x = _ultima_ventana(precios, scaler, WINDOW_SIZE)
        precio_predicho = float(modelo.predict(x, scaler=scaler)[0])

    cambio_pct = (precio_predicho - precio_actual) / precio_actual * 100
    if abs(cambio_pct) < 1e-9:
        direccion = "igual"
    else:
        direccion = "sube" if cambio_pct > 0 else "baja"

    fecha_ultimo = fechas[-1]
    fecha_pred = (dt.date.fromisoformat(fecha_ultimo) + dt.timedelta(days=1)).isoformat()

    os.makedirs(RESULTS_FIG_DIR, exist_ok=True)
    ruta_fig = os.path.join(RESULTS_FIG_DIR, f"prediccion_{nombre}.png")
    try:
        _graficar_prediccion(
            nombre, fechas, precios, precio_predicho, cambio_pct, fecha_pred, dias, ruta_fig
        )
        _log(f"Figura: {ruta_fig}")
    except Exception as exc:  # noqa: BLE001
        _log(f"Aviso al generar la figura: {exc}")

    return {
        "modelo": modelo_nombre,
        "objetivo": objetivo,
        "semilla": semilla,
        "fecha_ultimo_dato": fecha_ultimo,
        "fecha_prediccion": fecha_pred,
        "precio_actual": round(precio_actual, 4),
        "precio_predicho": round(precio_predicho, 4),
        "cambio_pct": round(cambio_pct, 4),
        "direccion": direccion,
        "metricas_test": _metricas_test(modelo_nombre, objetivo, semilla),
    }


def main():
    parser = argparse.ArgumentParser(description="Predice el cierre del dia siguiente.")
    parser.add_argument("--modelo", required=True, choices=list(REGISTRY.keys()))
    parser.add_argument("--objetivo", default="precio", choices=["precio", "retorno"])
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument(
        "--dias",
        type=int,
        default=90,
        help="Dias de historial que se muestran en la grafica (por defecto 90).",
    )
    args = parser.parse_args()

    if args.objetivo == "retorno" and args.modelo not in ("lstm", "gru"):
        raise SystemExit("El objetivo 'retorno' solo aplica a lstm y gru.")

    _log(f"Prediciendo manana con {args.modelo.upper()} ({args.objetivo}, seed {args.semilla})...")
    print(
        json.dumps(
            predecir(args.modelo, args.objetivo, args.semilla, args.dias),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
