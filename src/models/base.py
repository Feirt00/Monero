from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Método para entrenar el modelo."""
        pass

    @abstractmethod
    def predict(self, X):
        """Método para generar predicciones."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Método para guardar el modelo entrenado."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Método para cargar un modelo previamente guardado."""
        pass    