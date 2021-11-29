import sys
sys.path.append("..") 
import os
from src.get_data import read_params, get_data
import argparse
import pickle

def load_and_save(config_path):
    config = read_params(config_path)
    sem_data = get_data(config_path)
    raw_data_path = config["load_data"]["raw_dataset"]
    pickle.dump(sem_data, open(raw_data_path, 'wb'))


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)