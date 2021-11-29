from src.get_data import read_params
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import json
import numpy as np
from prediction_service.prediction import predict


def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config["mlflow_config"]

    model_name = mlflow_config["registered_model_name"]

    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    runs = mlflow.search_runs(experiment_ids=2)
    highest = runs["metrics.val_acc"].sort_values(ascending=True)[0]
    highest_run_id = runs[runs["metrics.val_acc"] == highest]["run_id"][0]

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)

        if mv["run_id"] == highest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )

    loaded_model = mlflow.pyfunc.load_model(logged_model)

    model_path = config["webapp_model_dir"]  # "prediction_service/model"

    data_request = np.asarray(json.load(open('/Users/Ahmed/Desktop/myfile.json',)))
    prediction = loaded_model.predict(data_request).tolist()
    print(prediction)
    #mlflow.keras.save_model(loaded_model,model_path)
    #cloudpickle.dump(loaded_model, open(model_path,'wb'))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)