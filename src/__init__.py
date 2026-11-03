from src.models.naive import NaiveModel
from src.models.moving_average import MovingAverageModel
from src.models.arima import ARIMAModel
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel

REGISTRY = {
    "naive": NaiveModel,
     "moving_average": MovingAverageModel,
     "arima": ARIMAModel,
     "lstm": LSTMModel,
     "gru": GRUModel,
}