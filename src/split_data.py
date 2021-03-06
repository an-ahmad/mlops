import sys
sys.path.append("..") 
import os
import argparse
from sklearn.model_selection import train_test_split
from src.get_data import read_params
import pickle
from imblearn.under_sampling import RandomUnderSampler
import numpy

def reshape_image(images):
    '''For reshaping images to desired size'''
    return numpy.asarray([images[i].reshape([96,96]) for i in range(len(images))])

def undersampling(X,y):
    '''For dealing with imbalanced data '''
    undersample = RandomUnderSampler(sampling_strategy='auto')
    X_over, y_over = undersample.fit_resample(X.reshape(X.shape[0],-1), y)
    return X_over , y_over 

def split_and_saved_data(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset"]
    split_ratio = config["split_data"]
    random_state = config["base"]["random_state"]
    split_ratio["test_size"]

    sem_imgs = pickle.load(open(raw_data_path, 'rb'))
    X = sem_imgs[0]
    y = sem_imgs[1]
    X , y = undersampling(X,y)
    X = X.reshape(X.shape[0],96,96,3)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y,\
            train_size=split_ratio["train_size"], test_size=split_ratio["test_size"], random_state=random_state)
    pickle.dump([X_train,y_train], open(train_data_path, 'wb'))
    pickle.dump([X_val, y_val], open(test_data_path, 'wb'))


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)