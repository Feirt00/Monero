import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.models.base import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, units=100, dropout=0.2, lr=0.001):
        super().__init__()
        self.name = "lstm"
        self.units = units
        self.dropout = dropout
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        """Construye la arquitectura LSTM de 2 capas con Dropout."""
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(30, 1)),
            Dropout(self.dropout),
            LSTM(self.units, return_sequences=False),
            Dropout(self.dropout),
            Dense(1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mse',
            metrics=['mae']
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Entrena el modelo LSTM aplicando Early Stopping."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if y_val is not None else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def predict(self, X, scaler=None):
        """Genera predicciones escaladas y las desescala a USD si se proporciona el scaler."""
        preds_scaled = self.model.predict(X, verbose=0).reshape(-1, 1)
        
        if scaler is not None:
            preds_usd = scaler.inverse_transform(preds_scaled)
            return preds_usd.flatten()
            
        return preds_scaled.flatten()

    def save(self, filepath):
        """Guarda el modelo entrenado en disco."""
        self.model.save(filepath)

    def load(self, filepath):
        """Carga el modelo entrenado desde disco."""
        self.model = load_model(filepath)