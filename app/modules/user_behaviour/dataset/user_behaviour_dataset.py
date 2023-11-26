import pandas as pd
import ast

from app.common.entities.dataset import Dataset
from app.modules.user_behaviour.dataset.utils import generate_behaviour_mock_dataset

class UserBehaviourDataset(Dataset):
  """Dataset for user behaviour."""

  def __init__(self, filename: str = None, dataframe: pd.DataFrame = None):
    """Initialize."""
    super().__init__(filename=filename, dataframe=dataframe)

  def populate(self):
    """Populate the dataset."""
    self._data = generate_behaviour_mock_dataset()
  
  def getTrainHyperparams(self):
    """Get hyperparameters for training."""
    return self._data.drop(columns=['plugged_time_minutes'])
  
  def getTestHyperparams(self):
    """Get hyperparameters for testing."""
    return self._data.drop(columns=['plugged_time_minutes'])

  def getTrainLabels(self):
    """Get labels for training."""
    return self._data['plugged_time_minutes']