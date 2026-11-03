# segunclaude.md

> **Qué es este archivo:** resumen completo de la conversación de trabajo sobre el proyecto final de Sistemas Inteligentes (predicción del precio de Monero / XMR). Recoge hallazgos, verificaciones hechas en la máquina, decisiones tomadas y lo que queda pendiente. Sirve como bitácora para no repetir el análisis y para justificar decisiones ante la profesora.
>
> **Fecha del análisis:** 30 de octubre de 2026
> **Carpeta del proyecto:** `C:\Aplicaciones\Sistemas Inteligentes\proyecto final`

---

## 0. Estado inicial de la carpeta

Solo había **2 archivos**:

| Archivo | Tamaño | Contenido |
|---|---|---|
| `Propuesta_Monero_IEEE (1) (2).docx` | 28 KB | Artículo IEEE de Fase 1 (Ayala, Díaz, Sánchez — Ing. de Sistemas, U. de Ibagué) |
| `rnn_lstm_gru_y_finance (2).ipynb` | 1.4 MB | Laboratorio guiado de 51 celdas: RNN vs LSTM vs GRU con `yfinance` |

---

## 1. Qué dice el documento (.docx)

**Título:** *Predicción del precio de la criptomoneda Monero mediante modelos de series de tiempo y aprendizaje automático.*

**Estructura:** Resumen · I. Introducción · II. Planteamiento del problema · III. Objetivos · IV. Estado del arte · V. Comparación de propuestas · VI. Datos y variables · VII. Metodología preliminar · VIII. Alcance y limitaciones · IX. Conclusiones · Referencias (9).

**Pregunta de investigación:** ¿una LSTM predice mejor el precio de cierre diario de Monero que modelos base como la media móvil, ARIMA o la regresión lineal?

**Metodología (§VII):**
- Partición **cronológica** (sin shuffle), normalización **MinMaxScaler ajustada solo con train**.
- Modelos base: **media móvil, regresión lineal, ARIMA**.
- Modelo principal: **LSTM** (GRU como alternativa), configuración de [5]: 2 capas × 100 unidades, dropout 0.2, Adam, lotes de 32, 20 épocas con parada temprana.
- Ventana de ejemplo: **30 días** para predecir el día 31.
- Métricas: **MAE, RMSE, MAPE** + **% de acierto de dirección**.
- Referencia [9] cita `TimeSeriesSplit`.

---

## 2. Decisión: la regresión lineal se elimina

La profesora indicó que **la regresión lineal no es necesaria**. El argumento técnico es sólido: la regresión lineal **no modela dependencia temporal** (no es un modelo de series de tiempo) y es redundante frente a media móvil + ARIMA.

**Riesgo detectado:** aparece en **4 lugares** del documento. Si se quita de unos y no de otros, el documento queda contradiciéndose.

| # | Ubicación | Texto actual | Acción |
|---|---|---|---|
| 1 | **Resumen** (párrafo inicial) | "…frente a modelos base como la media móvil, la regresión lineal y ARIMA" | quitar "la regresión lineal y" |
| 2 | **§II.B Pregunta de investigación** | "…que modelos base como la media móvil, ARIMA o la regresión lineal?" | quitar "o la regresión lineal" |
| 3 | **§III.B Objetivo específico 4** | "Construir modelos base, como media móvil o regresión lineal" | → "Construir modelos base, como media móvil y ARIMA" |
| 4 | **§VII.C Modelos** | "Los modelos base serán la media móvil, la regresión lineal y ARIMA" | → "…la media móvil y ARIMA" |

**Resultado:** los modelos base quedan **media móvil + ARIMA** (+ el baseline trivial "mañana = hoy", que no está en el documento pero sí en el código).

---

## 3. Verificación del notebook (celda por celda)

### ✅ Lo que sí está implementado y funciona

- **LSTM, GRU y RNN simple** con la misma arquitectura: 1 capa de 32 unidades + `Dense(1)`, Adam, MSE, `EarlyStopping(patience=8)`.
- **Baseline "mañana = hoy"** bien construido (`X_test[:, -1, 0]`).
- **Partición temporal 80/20** sin shuffle y **MinMaxScaler ajustado solo con train** → sin data leakage. Correcto.
- **RMSE y MAE** con `inverse_transform` a precios reales.
- **Ventanas deslizantes de 90 días**.
- **Extras no exigidos por el documento** (suman, pero no son el experimento del artículo):
  - Mejora 1: **retornos** como objetivo.
  - Mejora 2: **multivariado** (Close, High, Low, Volume).
  - Mejora 3: **dos capas recurrentes**.
  - Mejora 4: **3 semillas** → demostró que una sola corrida no concluye nada.
  - Paso 9: **descomposición STL** causal como entrada.
- **Conclusiones escritas con números reales** y limitaciones honestas.

### ❌ Lo que falta para cumplir el documento

| Requisito del documento | Estado en el notebook |
|---|---|
| **Media móvil** como baseline | ❌ no existe |
| **ARIMA** | ❌ no existe (`statsmodels` solo se usa para STL) |
| **MAPE** | ❌ nunca se calcula (solo RMSE/MAE) |
| **Acierto de dirección** (sube/baja) | ❌ no se plantea clasificación |
| **Datos de Monero (XMR)** | ❌ el notebook usa **NVDA**; la única cripto es BTC-USD |
| Variables del doc (% cambio, RSI, MACD, spreads) | ❌ solo Close/High/Low/Volume |
| `TimeSeriesSplit` (citado en §VII.B, ref. [9]) | ❌ se cita pero no se usa |
| Hiperparámetros de [5] (2 capas × 100, dropout 0.2, 20 épocas) | ❌ el código usa 1 capa × 32, sin dropout, 50 épocas |

> **Conclusión:** el notebook es un laboratorio docente muy completo y bien redactado, pero **no es todavía el experimento del artículo**. Le falta el núcleo comparativo: XMR + media móvil + ARIMA + MAPE + dirección. Además, **documento y código no coinciden** en arquitectura y ventana.

### Resultados reales del notebook (para referencia)

- Ninguna red superó al baseline: **baseline RMSE 4.189 / MAE 3.162** en NVDA.
- Con precio como objetivo: GRU 10.355 · RNN 13.033 · LSTM 55.854.
- Con retornos: ~4.30–4.37 (se acercan al baseline, pero no lo superan).
- 3 semillas: la GRU fue la mejor y más estable (7.434 ± 0.662); la RNN varió de 26 a 58.
- STL **no** mejoró a las redes (GRU 4.286 → 4.363).
- **Causa principal identificada:** los precios del test superaban el rango visto en entrenamiento → el escalador nunca vio esos valores y las redes se quedan cortas.

---

## 4. Decisión: fuente de datos = `yfinance` (`XMR-USD`)

### Ficha técnica verificada en la máquina (30/10/2026)

| Dato | Valor |
|---|---|
| Ticker | `XMR-USD` |
| Rango | 2017-11-09 → **2026-10-30** (se extiende sola en cada ejecución) |
| Filas | **3.278** velas diarias |
| Huecos | **0** (cripto cotiza 7 días a la semana) |
| Columnas | Open, High, Low, Close, Volume |
| Precio mín / máx | **33.01** / **713.44** USD |
| Split 80/20 | train **2.622** (2017-11 → ≈2025-01) · test **656** (≈2025-01 → 2026-10) |

Últimas velas devueltas en la prueba:

```
2026-10-26  596.84
2026-10-27  605.94
2026-10-28  598.12
2026-10-29  601.96
2026-10-30  587.78   ← vela del día
Precio actual: 587.78 USD
```

### Fuentes comparadas (probadas en vivo, no de memoria)

| Fuente | Prueba real | Veredicto |
|---|---|---|
| **yfinance `XMR-USD`** | 3.278 filas, 2017-11-09 → 2026-10-30, 0 nulos, OHLCV | ✅ **elegida** |
| **CoinGecko (API gratis)** | 366 puntos, solo desde 2025-10-31 | ⚠️ límite de **365 días**; histórico completo requiere plan Pro |
| **Kraken (API pública)** | 721 velas, 2024-11-09 → 2026-10-30 | ⚠️ solo ~2 años por request |
| **TradingView (export CSV)** | manual, símbolos `CRYPTO:XMRUSD`, `KRAKEN:XMRUSD`, `BITFINEX:XMRUSD`, `POLONIEX:XMRUSD` | ⚠️ foto fija, no automatizable; sirve para validar |
| **Investing.com** | CSV con rango personalizado, OHLC + volumen + % var | ⚠️ manual, ToS restringe scraping |
| **Bitget / CryptoStruct** | CSV / tick data | ⚠️ solo si se pidiera intradía o HFT |
| **Kaggle** | CSV estáticos | ⚠️ suelen estar desactualizados |

### Aclaración importante: histórico ≠ tiempo real

- **Para entrenar** se necesita historia (2017 → ayer). Que el archivo esté congelado no afecta: los datos de 2018 no cambian.
- **El modelo no aprende del "ahora"**. El tiempo real solo sirve para *predecir hoy/mañana* con el modelo ya entrenado.
- `yfinance` **se actualiza en cada ejecución** (trae la vela del día sin descargar nada a mano) → por eso es mejor que un CSV de TradingView, que se queda viejo al día siguiente.
- Si en Fase 2 se quisiera intradía: Kraken WebSocket, `ccxt` o polling a CoinGecko. **Pero eso cambia todo el modelo** (ventanas de minutos) y ya no sería lo que dice el documento.

### Regla adoptada

**Una sola fuente para entrenar y evaluar. No mezclar** `yfinance` con TradingView/Kraken/CoinGecko: los cierres difieren 1–3% y el volumen difiere muchísimo (cada exchange reporta el suyo). Mezclar fuentes invalida la comparación entre modelos.

---

## 5. Lista de modelos que toca entrenar

### Bloque 1 — Referencias (no entrenan)

| # | Modelo | Definición | Salida |
|---|---|---|---|
| 1 | **Persistencia (naive)** "mañana = hoy" | `ŷ(t+1) = y(t)` | precio |
| 2 | **Media móvil (7 días)** | promedio de los últimos 7 cierres | precio |

### Bloque 2 — Modelo estadístico

| # | Modelo | Definición | Librería |
|---|---|---|---|
| 3 | **ARIMA(p,d,q)** | orden por **AIC** sobre train; pronóstico 1 paso con parámetros fijos | `statsmodels` ⚠️ falta instalar |

### Bloque 3 — Redes recurrentes (núcleo del artículo)

| # | Modelo | Arquitectura | Entrena |
|---|---|---|---|
| 4 | **LSTM** | 1 capa × 32 (notebook) o 2 × 100 + dropout 0.2 (doc/[5]) | ✅ |
| 5 | **GRU** | idéntica, solo cambia la capa | ✅ |
| (6) | *RNN simple* | opcional, ya existe en el notebook | ✅ |

### Variantes por red

| Variante | Objetivo | Motivo |
|---|---|---|
| **A** | Precio de cierre (nivel) | lo que dice el documento literalmente |
| **B** | **Retorno logarítmico** → se reconstruye el precio | evita el problema de escala 250 → 713 |

### Total de corridas

```
2 baselines + 1 ARIMA + (2 redes × 2 variantes × 3 semillas = 12) = 15 corridas
```

### Protocolo común (obligatorio)

- Mismo test: **los mismos 656 días** para todos.
- Sin shuffle, split cronológico.
- Scaler ajustado **solo con train**.
- Métricas: **MAE, RMSE, MAPE, acierto de dirección** (+ tiempo de entrenamiento como extra).
- Semillas `[0, 1, 2]` y reportar **media ± desviación**.

---

## 6. Estructura de código acordada (`.py`, no notebook)

Ver **`ESTRUCTURA.md`** (mismo folder) para el detalle. Resumen:

```
proyecto final/
├── config.py                  # ventana, split, semillas, rutas
├── data/
│   ├── download.py            # yfinance → data/raw/xmr_usd_diario.csv
│   └── prepare.py             # split congelado + scaler + ventanas → dataset.npz
├── models/
│   ├── base.py                # interfaz fit()/predict() + @register
│   ├── naive.py               # 1
│   ├── moving_average.py      # 2
│   ├── arima.py               # 3
│   ├── lstm.py                # 4
│   └── gru.py                 # 5
├── evaluation/
│   ├── metrics.py             # MAE, RMSE, MAPE, dirección
│   └── compare.py             # corre los 5 → comparacion.csv
├── train.py                   # entrena UN modelo (CLI)
├── predict.py                 # predice mañana → JSON (lo consume el frontend)
├── models/                    # artefactos (.keras, .pkl) — no va a git
└── results/{metrics,predictions,figures}/
```

**Razones de diseño:**
1. **Un `.py` por modelo**, todos con la misma interfaz → el compañero del frontend conecta cada uno sin acoplarse.
2. **`predict.py` devuelve JSON** con `precio_predicho`, `direccion`, `cambio_pct` y métricas → el frontend **no importa TensorFlow**.
3. **Split congelado en `dataset.npz`** → los 5 modelos se evalúan sobre los mismos 656 días.
4. Cuando se quiera API, ese JSON se envuelve en FastAPI sin tocar el entrenamiento.

### Contrato JSON para el frontend

```json
{
  "modelo": "lstm",
  "objetivo": "retorno",
  "fecha_ultimo_dato": "2026-10-30",
  "fecha_prediccion": "2026-10-31",
  "precio_actual": 587.78,
  "precio_predicho": 592.10,
  "cambio_pct": 0.73,
  "direccion": "sube",
  "metricas_test": { "MAE": 12.30, "RMSE": 17.81, "MAPE": 2.79, "acierto_direccion": 51.4 }
}
```

---

## 7. Advertencias técnicas (riesgos identificados)

1. **Problema de escala:** el train termina cerca de 250 USD y el test llega a 713 USD. Si se predice el **precio en nivel** con MinMaxScaler, las redes fallarán por la misma razón que en NVDA. → **Predecir retorno** y reconstruir el precio.
2. **Una sola corrida no concluye nada:** el notebook demostró variación enorme entre semillas (RNN de 26 a 58 de RMSE). → mínimo 3 semillas y reportar desviación.
3. **Cripto ≠ acciones:** cotiza 7/7. La media móvil debe ser de **7 días** y el periodo de STL debe ser **7**, no 21 (el notebook usa lógica de acciones).
4. **MAPE** es válido aquí porque todos los precios son > 0.
5. **Documento y código deben coincidir:** hoy el doc dice 2 capas × 100 + dropout 0.2 y 30 días; el notebook usa 1 × 32 y 90 días. Hay que alinear una de las dos versiones.
6. **`statsmodels` no está instalado** en el venv de trabajo (`C:\Aplicaciones\Sistemas Inteligentes\taller\.venv`) → necesario para ARIMA (y para el STL del notebook).

### Estado del entorno (verificado)

| Paquete | Estado |
|---|---|
| `yfinance` 1.7.0 | ✅ |
| `pandas` 3.0.6 · `numpy` 2.5.4 | ✅ |
| `scikit-learn` 1.9.1 | ✅ |
| `tensorflow` 2.21.0 · `keras` 3.15.1 | ✅ |
| `matplotlib` 3.11.2 | ✅ |
| **`statsmodels`** | ❌ **falta** |
| `pmdarima` (opcional, auto-ARIMA) | ❌ falta |

---

## 8. Pendientes (actualizado 2026-11-03)

Completado en la sesión del 3 de noviembre:

- [x] **Bug de alineación corregido:** `y_test_real`, `base_precio`, `ret_test_real` y `test_dates` ahora tienen 656 filas, igual que `X_test` (antes sobraban/desfasaban 30 días y el naive reportaba MAE 54.12 en vez de 9.69).
- [x] **Código:** objetivo `retorno` logarítmico implementado en `prepare.py` y `train.py` (precio reconstruido como `base · e^r`).
- [x] **Código:** los 5 modelos entrenados y evaluados; `compare.py` agrega media ± desviación.
- [x] **Código:** `predict.py` devuelve el JSON del contrato con el precio de mañana.
- [x] **Código:** imports pesados movidos dentro de las funciones (los tests ya no cargan TensorFlow).
- [x] **Extras:** `requirements.txt`, `README.md`, `results/predictions/`, `results/figures/`, tests de alineación.
- [x] **Documento (preparado):** versión editada sin "regresión lineal", con fuente yfinance y epochs alineados. Falta reemplazar el .docx original (Word lo tiene abierto).

Pendiente real:

- [ ] Reemplazar `Propuesta_Monero_IEEE (1) (2).docx` por la versión editada (cerrar Word primero).
- [ ] Escribir el análisis de resultados (qué modelo ganó y por qué) en el artículo/informe final.


---

## 9. Comandos de referencia

```bash
# entorno
pip install yfinance statsmodels scikit-learn tensorflow matplotlib pandas numpy

# datos
python -m src.data.download
python -m src.data.prepare

# modelos uno por uno
python -m src.train --modelo naive          --objetivo precio
python -m src.train --modelo moving_average --objetivo precio
python -m src.train --modelo arima          --objetivo precio
python -m src.train --modelo lstm           --objetivo retorno --semilla 0
python -m src.train --modelo gru            --objetivo retorno --semilla 0

# comparación y predicción
python -m src.evaluation.compare
python -m src.predict --modelo lstm --objetivo retorno
```

```python
# descarga mínima (equivale a download.py)
import yfinance as yf
df = yf.download("XMR-USD", start="2017-11-09", auto_adjust=True, progress=False)
df.columns = df.columns.get_level_values(0)
df.to_csv("data/raw/xmr_usd_diario.csv")
```

---

*Documento generado a partir de la conversación de trabajo. Todas las cifras de datos (filas, fechas, precios, límites de APIs) fueron verificadas ejecutando consultas reales el 30/10/2026, no tomadas de memoria.*
