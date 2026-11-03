import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from src.models.base import BaseModel

class ARIMAModel(BaseModel):
    def __init__(self, order=None):
        super().__init__()
        self.order = order  # Si es None, se busca por AIC
        self.best_order = (1, 1, 1)  # Valor por defecto defensivo
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Ajusta el modelo ARIMA encontrando el mejor orden (p,d,q) por AIC 
        sobre los datos de entrenamiento, o usando el orden predefinido.
        """
        # Extraer la serie temporal de entrenamiento
        if len(X_train.shape) == 3:
            train_series = y_train.flatten() if y_train is not None else X_train[:, -1, 0]
        else:
            train_series = X_train.flatten()

        self.history = list(train_series)

        if self.order is not None:
            self.best_order = self.order
        else:
            # Búsqueda simple por AIC (Grid Search liviano para p, d, q)
            best_aic = float("inf")
            best_ord = (1, 1, 1)
            
            print("Buscando el mejor orden (p,d,q) por AIC para ARIMA...")
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = ARIMA(self.history, order=(p, d, q))
                            results = model.fit()
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_ord = (p, d, q)
                        except Exception:
                            continue
            self.best_order = best_ord
            print(f"¡Mejor orden ARIMA encontrado por AIC: {self.best_order} (AIC: {best_aic:.2f})!")

    def predict(self, X, scaler=None):
        """
        Pronóstico iterativo de 1 paso para el conjunto de test y desescalado a USD.
        """
        predictions = []
        current_history = list(self.history)

        if len(X.shape) == 3:
            test_sequences = X[:, :, 0]
        else:
            test_sequences = X

        for seq in test_sequences:
            try:
                model = ARIMA(current_history, order=self.best_order)
                model_fit = model.fit()
                pred = model_fit.forecast(steps=1)[0]
            except Exception:
                # Fallback defensivo si ARIMA diverge en algún punto
                pred = current_history[-1]

            predictions.append(pred)
            # Actualizamos el historial con el valor real de la secuencia para mantener el paso a paso
            current_history.append(seq[-1])

        preds_scaled = np.array(predictions).reshape(-1, 1)

        # Desescalar a USD si se proporciona el scaler
        if scaler is not None:
            preds_usd = scaler.inverse_transform(preds_scaled)
            return preds_usd.flatten()

        return preds_scaled.flatten()

    def save(self, filepath):
        pass  # Los baselines no requieren guardar artefactos pesados

    def load(self, filepath):
        pass