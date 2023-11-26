import os
import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from app.common.entities import Model
from app.common.helpers import get_config

class RecurrentRegressionModel(Model):
  """Recurrent regression model."""

  def __init__(self, *args, **kwargs):
    """Initialize the model."""
    config = get_config()
    self.input_dim = config.getint('RecurrentRegressionModel', 'InputDim')
    self.num_epochs = config.getint('RecurrentRegressionModel', 'NumEpochs')
    super().__init__(*args, **kwargs)

  def _build(self):
    """Build the model."""
    self._model = Sequential()
    self._model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(1, self.input_dim)))
    self._model.add(Dropout(0.2))
    self._model.add(LSTM(units=50, activation='relu', return_sequences=True))
    self._model.add(Dropout(0.2))
    self._model.add(LSTM(units=50, activation='relu'))
    self._model.add(Dense(1))
    self._model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

  def train(self, dataset):
    """Train the model."""
    train_data = dataset.getTrainHyperparams().values.reshape(-1, 1, self.input_dim)
    train_labels = dataset.getTrainLabels().values.reshape(-1, 1)
    self._model.fit(
      train_data, train_labels,
      epochs=self.num_epochs, batch_size=1, verbose=1,
      validation_split=0.2
    )

  def predict(self, dataset):
    """Predict."""
    return self._model.predict(
      dataset.getTestHyperparams().values.reshape(-1, 1, self.input_dim)
    ).flatten().tolist()
