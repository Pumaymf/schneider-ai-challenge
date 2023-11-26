import pandas as pd

from app.common.entities.dataset import Dataset
from app.modules.discharge_capacity.dataset.utils import generate_discharge_mock_dataset

class DischargeCapacityDataset(Dataset):
  """Dataset for discharge capacity."""

  def __init__(self, filename: str = None, dataframe: pd.DataFrame = None):
    """Initialize."""
    super().__init__(filename=filename, dataframe=dataframe)

  def populate(self):
    """Populate the dataset."""
    self._data = generate_discharge_mock_dataset()

  def getTrainHyperparams(self):
    """Get hyperparameters for training."""
    return self._data[['day_of_week', 'week_of_year', 'usable_capacity']]
  
  def getTestHyperparams(self):
    """Get hyperparameters for testing."""
    return self._data[['day_of_week', 'week_of_year', 'usable_capacity']]

  def getTrainLabels(self):
    """Get labels for training."""
    return self._data['discharged']