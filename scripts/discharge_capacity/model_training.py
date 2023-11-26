import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse

from app.modules.discharge_capacity.dataset import DischargeCapacityDataset
from app.modules.discharge_capacity.model import RecurrentRegressionModel

def main(args):
  dataset = DischargeCapacityDataset(filename=args.dataset)
  model = RecurrentRegressionModel()
  model.train(dataset)
  model.save(filename=args.model_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Train the discharge capacity model.'
  )
  parser.add_argument(
    '--dataset', type=str,
    default=os.path.join(os.path.dirname(__file__), '../../data/discharge_capacity/train.csv'),
    help='The path to the processed dataset.'
  )
  parser.add_argument(
    '--model_file', type=str,
    default=os.path.join(os.path.dirname(__file__), '../../model/discharge_capacity/recurrent_regression_model.pkl'),
    help='The path where the trained model will be stored.'
  )
  args = parser.parse_args()
  main(args)
