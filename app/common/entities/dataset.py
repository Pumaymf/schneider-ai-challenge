from abc import ABC, abstractmethod

import pandas as pd

class Dataset(ABC):
  """Abstract class for operation."""

  def __init__(self, filename: str = None, dataframe: pd.DataFrame = None):
    """Initialize."""
    if filename is not None:
      self._data = pd.read_csv(filename, index_col=0)
    elif dataframe is not None:
      self._data = dataframe
    else:
      self._data = None
    
  def save(self, filename: str) -> None:
    self._data.to_csv(filename, index=True)

  @abstractmethod
  def populate(self):
    """Populate the dataset."""
    pass
