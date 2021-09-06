import os
import warnings
import sys
import numpy as np
from get_data import read_params
import argparse
import joblib
import json
from tensorflow.keras import models
from tensorflow.keras import layers
import pickle


def build_toy_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), input_shape=(96, 96, 3),
                            kernel_initializer='he_normal', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def eval_metrics(actual, pred,model):
    val_acc = model.evaluate(actual,pred)
    return val_acc[-1]


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    #random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    epochs = config["estimators"]["cnn"]["params"]["epochs"]
    batch_size = config["estimators"]["cnn"]["params"]["batch_size"]

    train = pickle.load(open(train_data_path, 'rb'))
    test = pickle.load(open(test_data_path, 'rb'))
    X_train = np.asarray(train[0])
    y_train = np.asarray(train[1])
    X_val = np.asarray(test[0])
    y_val = np.asarray(test[1])

    model = build_toy_model()
    model.fit(X_train, y_train, epochs=1, batch_size=32,
              verbose=1, validation_data=(X_val, y_val)) #, callbacks=[save])
    print(y_val.shape, y_train.shape)

    val_accuracy = eval_metrics(X_val, y_val,model)


    print("  val_acc: {}".format(val_accuracy) )

    #####################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "val_acc": val_accuracy,
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "epochs": epochs,
            "batch_size": batch_size,
        }
        json.dump(params, f, indent=4)
    #####################################################

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.h5")

    model.save(model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)