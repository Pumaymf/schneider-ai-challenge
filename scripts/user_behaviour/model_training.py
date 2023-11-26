import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse

from app.modules.user_behaviour.dataset import UserBehaviourDataset
from app.modules.user_behaviour.model import RecurrentRegressionWindowModel

def main(args):
  dataset = UserBehaviourDataset(filename=args.dataset)
  model = RecurrentRegressionWindowModel()
  model.train(dataset)
  model.save(filename=args.model_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Train the user behaviour model.'
  )
  parser.add_argument(
    '--dataset', type=str,
    default=os.path.join(os.path.dirname(__file__), '../../data/user_behaviour/train.csv'),
    help='The path to the processed dataset.'
  )
  parser.add_argument(
    '--model_file', type=str,
    default=os.path.join(os.path.dirname(__file__), '../../model/user_behaviour/recurrent_regression_model.pkl'),
    help='The path where the trained model will be stored.'
  )
  args = parser.parse_args()
  main(args)
