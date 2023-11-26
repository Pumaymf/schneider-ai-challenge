import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import argparse

from app.modules.discharge_capacity.dataset import DischargeCapacityDataset

def main(args):
  dataset = DischargeCapacityDataset()
  dataset.populate()
  dataset.save(filename=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/discharge_capacity/train.csv")))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Populate the discharge capacity dataset.'
  )
  args = parser.parse_args()
  main(args)
