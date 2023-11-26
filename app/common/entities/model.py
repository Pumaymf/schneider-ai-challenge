import pickle

from abc import abstractmethod

class Model():
  """Wrapper class for model operation."""

  def __init__(self, filename: str = None):
    """Initialize."""
    if filename is not None:
      self.load(filename)
    else:
      self._build()

  @abstractmethod
  def train(self):
    """Fit the model."""
    pass

  @abstractmethod
  def predict(self):
    """Predict."""
    pass

  def load(self, filename: str) -> None:
    self._model = pickle.load(open(filename, 'rb'))
    pass

  def save(self, filename: str) -> None:
    pickle.dump(self._model, open(filename, 'wb'))
    pass

  @abstractmethod
  def _build(self) -> None:
    """Build the model."""
    pass