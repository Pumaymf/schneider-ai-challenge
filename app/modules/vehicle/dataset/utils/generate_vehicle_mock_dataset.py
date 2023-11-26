# dataset of vehicles has columns:
#   date
#   vehicle_id
#   vehicle_usable_capacity
#   vehicle_discharge
#   vehicle_plugged_time

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from itertools import product

from app.common.helpers import get_config;

def generate_vehicle_mock_dataset() -> pd.DataFrame:
  config = get_config()
  num_vehicles = config.getint('MockDataset', 'NumVehicles')
  start_date = datetime.strptime(config.get('MockDataset', 'StartDate'), '%Y-%m-%d')
  end_date = datetime.strptime(config.get('MockDataset', 'EndDate'), '%Y-%m-%d')
  min_battery_capacity = config.getint('MockDataset', 'MinBatteryCapacity')
  max_battery_capacity = config.getint('MockDataset', 'MaxBatteryCapacity')
  max_plugged_time = config.getint('MockDataset', 'MaxPluggedTime')

  np.random.seed(42)

  # Generate all possible combinations of dates and vehicle_ids
  dates = pd.date_range(start=start_date, end=end_date, freq='D')
  vehicle_ids = np.arange(0, num_vehicles)
  all_combinations = list(product(dates, vehicle_ids))

  # Create de dataframe
  df = pd.DataFrame(all_combinations, columns=['date', 'vehicle_id'])
  df['usable_capacity'] = np.random.randint(min_battery_capacity, max_battery_capacity, size=len(df))
  df['discharged'] = df['usable_capacity'] * np.random.random()
  df['plugged_time_minutes'] = np.random.randint(1, max_plugged_time, size=len(df))

  return df