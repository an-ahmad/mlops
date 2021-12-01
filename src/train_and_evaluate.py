import os
import warnings
import sys
sys.path.append("..") 
import numpy as np
from src.get_data import read_params
import argparse
from urllib.parse import urlparse
import json
from tensorflow.keras import models
from tensorflow.keras import layers
import pickle
import mlflow
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout


def build_model():
    model = models.Sequential()
    acti = 'relu'
    model.add(layers.Conv2D(64, (3, 3), input_shape=(96,96,3), 
              kernel_initializer='he_normal', activation=acti))
    model.add(BatchNormalization())
   
    model.add(layers.Conv2D(64, (3, 3), kernel_initializer='he_normal',activation=acti))
    model.add(BatchNormalization(axis= 2))
    model.add(layers.Conv2D(128, (3, 3), kernel_initializer='he_normal',activation=acti))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), kernel_initializer='he_normal',activation=acti))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(layers.Conv2D(128, (3, 3), kernel_initializer='he_normal',activation=acti))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), kernel_initializer='he_normal',activation=acti))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), kernel_initializer='he_normal',activation=acti))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(layers.Conv2D(256, (3, 3), kernel_initializer='he_normal',activation=acti))
    model.add(BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), kernel_initializer='he_normal',activation=acti))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(10, (1, 1), kernel_initializer='he_normal',activation=acti)) 
    model.add(BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(6, 6)))
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

    ################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    #mlflow.set_tracking_uri('http://127.0.0.1:5000')

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        model = build_model()
        model.fit(X_train, y_train, epochs=1, batch_size=32,
                  verbose=1, validation_data=(X_val, y_val)) #, callbacks=[save])
        print(y_val.shape, y_train.shape)

        val_accuracy = eval_metrics(X_val, y_val,model)

        print("  val_acc: {}".format(val_accuracy) )

        #####################################################
        # scores_file = config["reports"]["scores"]
        # params_file = config["reports"]["params"]
        #
        # with open(scores_file, "w") as f:
        #     scores = {
        #         "val_acc": val_accuracy,
        #     }
        #     json.dump(scores, f, indent=4)
        #
        # with open(params_file, "w") as f:
        #     params = {
        #         "epochs": epochs,
        #         "batch_size": batch_size,
        #     }
        #     json.dump(params, f, indent=4)
        #####################################################

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        mlflow.log_metric("val_acc", val_accuracy)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.keras.log_model(
                model,
                "model",
                registered_model_name= mlflow_config["registered_model_name"])
        else:
            mlflow.keras.load_model(model, "model")

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.h5")

        model.save(model_path)
        model.save(config['webapp_model_dir'])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)