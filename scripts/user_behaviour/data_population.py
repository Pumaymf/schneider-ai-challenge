import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse

from app.modules.user_behaviour.dataset import UserBehaviourDataset

def main(args):
  dataset = UserBehaviourDataset()
  dataset.populate()
  dataset.save(filename=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/user_behaviour/train.csv")))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Populate the user behaviour dataset.'
  )
  args = parser.parse_args()
  main(args)
