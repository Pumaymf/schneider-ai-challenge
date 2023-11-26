import numpy as np
import pandas as pd

from app.modules.vehicle.dataset.utils import generate_vehicle_mock_dataset

def generate_behaviour_mock_dataset() -> pd.DataFrame:
  vehicle_df = generate_vehicle_mock_dataset()
  user_behaviour_df = vehicle_df.drop(columns=['usable_capacity', 'discharged'])
  user_behaviour_df['last_30_days'] = user_behaviour_df.apply(
    get_last_30_days, axis=1, dataframe=user_behaviour_df
  )
  user_behaviour_df = pd.concat([user_behaviour_df, pd.DataFrame(user_behaviour_df['last_30_days'].tolist(), 
                                                                  columns=[f'day_{i+1}' for i in range(30)])], 
                                  axis=1)
  user_behaviour_df['day_of_week'] = user_behaviour_df['date'].dt.dayofweek
  user_behaviour_df['week_of_year'] = user_behaviour_df['date'].dt.isocalendar().week
  user_behaviour_df = user_behaviour_df.drop(columns=['date', 'vehicle_id', 'last_30_days'])

  return user_behaviour_df

def get_last_30_days(row, dataframe):
  filtered_data = dataframe[(dataframe['vehicle_id'] == row['vehicle_id']) & 
                            (dataframe['date'] >= row['date'] - pd.DateOffset(days=30)) &
                            (dataframe['date'] < row['date'])]['plugged_time_minutes'].tolist()
  if len(filtered_data) < 30:
    padded_data = np.pad(filtered_data, (30 - len(filtered_data), 0), mode='constant', constant_values=0)
  else:
    padded_data = filtered_data[-30:]
  return padded_data