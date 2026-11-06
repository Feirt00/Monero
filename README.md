# XMR-Forecast — Predicción del precio de Monero (XMR)

Proyecto final de Sistemas Inteligentes (Universidad de Ibagué). Compara cinco
modelos de series de tiempo para predecir el cierre diario de Monero (`XMR-USD`)
con un split cronológico congelado, métricas comunes y artefactos en disco para
que el frontend no dependa de TensorFlow.

**Pregunta:** ¿una LSTM predice mejor el cierre diario de Monero que la
persistencia, la media móvil, ARIMA o una GRU?

## Modelos comparados

| # | Modelo | Objetivo | Archivo |
|---|---|---|---|
| 1 | Persistencia ("mañana = hoy") | precio | `src/models/naive.py` |
| 2 | Media móvil de 7 días | precio | `src/models/moving_average.py` |
| 3 | ARIMA(p,d,q) por AIC, walk-forward 1 paso | precio | `src/models/arima.py` |
| 4 | LSTM 2×100 + dropout 0.2 | retorno logarítmico | `src/models/lstm.py` |
| 5 | GRU 2×100 + dropout 0.2 | retorno logarítmico | `src/models/gru.py` |

Protocolo: split cronológico 80/20 sin shuffle, `MinMaxScaler` ajustado **solo
con train**, ventana de 30 días, semillas `[0, 1, 2]` en las redes y métricas
MAE, RMSE, MAPE y % de acierto de dirección sobre **los mismos 657 días**.

## Requisitos

- Python 3.12
- Windows + PowerShell (los comandos de abajo)
- ~2 GB libres (TensorFlow es la dependencia más pesada)

## Instalación

Todo se ejecuta **desde la carpeta `proyecto final`**:

```powershell
cd "C:\Aplicaciones\Sistemas Inteligentes\proyecto final"

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activar el entorno es **opcional**; solo sirve para escribir `python` en vez de
la ruta completa:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

> Las líneas rojas con `Pandas4Warning` **no son errores**: `yfinance` y `pandas`
> las escriben por `stderr` y PowerShell las pinta como error. Se silencian con
> `$env:PYTHONWARNINGS = "ignore"`.

## Reproducir todo

Copia y pega todo el bloque para reproducir el proyecto de cero:

```powershell
cd "C:\Aplicaciones\Sistemas Inteligentes\proyecto final"
$env:PYTHONWARNINGS = "ignore"

# 1) Datos: descarga y actualiza XMR-USD hasta hoy
#    -> data/raw/xmr_usd_diario.csv
.\.venv\Scripts\python.exe -m src.data.download

# 2) Split congelado + escaladores + ventanas (precio y retorno)
#    -> data/processed/dataset.npz
.\.venv\Scripts\python.exe -m src.data.prepare

# 3) Baselines y modelo estadístico
.\.venv\Scripts\python.exe -m src.train --modelo naive          --objetivo precio
.\.venv\Scripts\python.exe -m src.train --modelo moving_average --objetivo precio
.\.venv\Scripts\python.exe -m src.train --modelo arima          --objetivo precio

# 4) Redes recurrentes (3 semillas cada una)
.\.venv\Scripts\python.exe -m src.train --modelo lstm --objetivo retorno --semilla 0
.\.venv\Scripts\python.exe -m src.train --modelo lstm --objetivo retorno --semilla 1
.\.venv\Scripts\python.exe -m src.train --modelo lstm --objetivo retorno --semilla 2
.\.venv\Scripts\python.exe -m src.train --modelo gru  --objetivo retorno --semilla 0
.\.venv\Scripts\python.exe -m src.train --modelo gru  --objetivo retorno --semilla 1
.\.venv\Scripts\python.exe -m src.train --modelo gru  --objetivo retorno --semilla 2

# 5) Tabla comparativa (media ± desviación) e imprime en consola
.\.venv\Scripts\python.exe -m src.evaluation.compare

# 6) Predicción de mañana: imprime JSON y guarda la gráfica
.\.venv\Scripts\python.exe -m src.predict --modelo lstm --objetivo retorno --semilla 0
# --dias N controla cuántos días de historial se dibujan (por defecto 90)

# 7) Pruebas
.\.venv\Scripts\python.exe -m pytest -q

# 8) Ver resultados
Get-Content .\results\metrics\comparacion.csv
Get-ChildItem .\results\figures
```

> Si ya tienes `data/processed/dataset.npz` y los modelos entrenados, puedes
> saltar directo al paso 5. Pero si vuelves a ejecutar `download` + `prepare`,
> **debes reentrenar**, porque el split congelado cambia y los artefactos viejos
> quedan desalineados.

## Salidas

```
models/<modelo>_<objetivo>_seed<k>.keras|.pkl   # artefactos (no van a git)
results/metrics/<modelo>_<objetivo>_seed<k>.json
results/metrics/comparacion.csv                  # media ± desviación por modelo
results/metrics/comparacion_detalle.csv          # una fila por corrida
results/predictions/<modelo>_<objetivo>_seed<k>_test.csv
results/figures/<modelo>_<objetivo>_seed<k>.png
results/figures/prediccion_<modelo>_<objetivo>_seed<k>.png   # historial + punto predicho
```

## Resultados (datos hasta 2026-11-06, test de 657 días)

| Modelo | Objetivo | MAE (USD) | RMSE (USD) | MAPE (%) | Acierto dirección (%) |
|---|---|---|---|---|---|
| Persistencia (naive) | precio | **9.75** | **15.09** | **2.67** | 1.5 |
| GRU 2×100 | retorno | 9.87 ± 0.10 | 15.29 ± 0.12 | 2.70 ± 0.03 | 51.0 ± 1.1 |
| LSTM 2×100 | retorno | 9.96 ± 0.08 | 15.37 ± 0.11 | 2.72 ± 0.02 | 48.1 ± 1.5 |
| ARIMA(2,0,1) | precio | 10.01 | 15.46 | 2.74 | 47.2 |
| Media móvil 7 días | precio | 15.22 | 25.04 | 4.09 | 54.6 |

Lectura: **ninguna red supera a la persistencia** en MAE/RMSE/MAPE. En esta
corrida la GRU queda levemente por delante de la LSTM, pero la diferencia entre
ambas es menor que su variación entre semillas. En dirección, la media móvil
acierta más (54.6 %) pero con un error de nivel mucho mayor; LSTM/GRU rondan el
50 %, es decir, cerca del azar. El 1.5 % del naive **no es comparable**: el naive
nunca predice subida/bajada (siempre dice "igual"), así que casi siempre falla la
dirección por definición.

Conclusión honesta: las redes **no** demuestran una ventaja predictiva sobre la
persistencia para el cierre diario de XMR. Un resultado negativo también es un
resultado válido y así se reporta.

## Contrato JSON para el frontend

`python -m src.predict --modelo lstm --objetivo retorno --semilla 0` imprime por
`stdout` (los logs y la ruta de la figura van a `stderr`):

```json
{
  "modelo": "lstm",
  "objetivo": "retorno",
  "semilla": 0,
  "fecha_ultimo_dato": "2026-11-06",
  "fecha_prediccion": "2026-11-07",
  "precio_actual": 619.47,
  "precio_predicho": 619.56,
  "cambio_pct": 0.01,
  "direccion": "sube",
  "metricas_test": {"MAE": 9.99, "RMSE": 15.42, "MAPE": 2.73, "acierto_direccion": 47.03}
}
```

Además guarda la gráfica del historial reciente más el punto predicho en
`results/figures/prediccion_<modelo>_<objetivo>_seed<k>.png`.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Cubren métricas, contrato `fit`/`predict`, ausencia de leakage y la alineación
1:1 entre ventanas, precios reales, fechas y base de retornos.

## Solución de problemas

| Síntoma | Causa / solución |
|---|---|
| `No module named 'yfinance'` (o `tensorflow`, `statsmodels`, `matplotlib`) | El entorno no tiene las dependencias: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| `No module named src.data.prepare` | No estás dentro de `proyecto final`, o falta el archivo. Verifica con `Get-ChildItem .\src\data`. |
| Texto rojo con `Pandas4Warning` / `NativeCommandError` | Son advertencias por `stderr`, no errores. Usa `$env:PYTHONWARNINGS = "ignore"`. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| El comando usa un Python equivocado | Usa la ruta explícita `.\.venv\Scripts\python.exe ...` en vez de `python ...` |
| `git push` rechazado (`fetch first`) | `git pull --rebase origin master` y luego `git push origin master` |

## Nota metodológica

Predecir el precio en nivel falla por escala (el train termina cerca de 250 USD y
el test llega a 713 USD). Por eso las redes se entrenan sobre el **retorno
logarítmico** y el precio se reconstruye como `precio_t = precio_{t-1} · e^{r_t}`;
las métricas se reportan siempre en USD para que los cinco modelos sean
comparables.

## Documentos

- [`Propuesta_Monero_IEEE (1) (2).docx`](Propuesta_Monero_IEEE%20(1)%20(2).docx): propuesta inicial en formato IEEE.
- [`Informe_Monero_Resultados.docx`](Informe_Monero_Resultados.docx): informe de implementación y resultados (6 páginas, 4 tablas y 3 figuras, mismo formato IEEE).

## Estructura

Ver [`ESTRUCTURA.md`](ESTRUCTURA.md) para el detalle de carpetas, interfaces y
reglas del proyecto.
