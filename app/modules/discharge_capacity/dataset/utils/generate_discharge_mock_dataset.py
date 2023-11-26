# dataset of vehicles has columns:
#   date
#   vehicle_id
#   vehicle_usable_capacity
#   vehicle_discharge
#   vehicle_plugged_time

# that will be transformed into (discharge_capacity):
#   date.day_of_week
#   date.week_of_year
#   weatherInfo
#   vehicle[
#       vehicle_usable_capacity
#       vehicle_recent_discharges (window)
#   ]


# that will be transformed into (user_behaviour):
#   date.day_of_week
#   date.week_of_year
#   weatherInfo
#   vehicle_plugged_time (window)

import pandas as pd

from app.modules.vehicle.dataset.utils import generate_vehicle_mock_dataset

def generate_discharge_mock_dataset() -> pd.DataFrame:
  vehicle_df = generate_vehicle_mock_dataset()
  discharge_capacity_df = vehicle_df.drop(columns=['plugged_time_minutes'])
  discharge_capacity_df['day_of_week'] = discharge_capacity_df['date'].dt.dayofweek
  discharge_capacity_df['week_of_year'] = discharge_capacity_df['date'].dt.isocalendar().week
  discharge_capacity_df = discharge_capacity_df.groupby(['date', 'day_of_week', 'week_of_year']).agg({
    'usable_capacity': 'sum',
    'discharged': 'sum'
  }).reset_index()

  return discharge_capacity_df
