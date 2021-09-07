import yaml
import os
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model

params_path = "params.yaml"


class NotInRange(Exception):
    def __init__(self, message="Invalid input, please input image of shape (96,96,3)"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    try:
        config = read_params(params_path)
        model_dir_path = config["webapp_model_dir"]
        model = load_model(os.path.join(model_dir_path))
        prediction = model.predict(data).tolist()
        return prediction

    except NotInRange:
        return "Unexpected result"


# def get_schema(schema_path=schema_path):
#     with open(schema_path) as json_file:
#         schema = json.load(json_file)
#     return schema


def validate_input(data_request):

    def _validate_values(val):

        if not val.shape == (96,96,3):
            raise NotInRange

    data_request = np.asarray(data_request)

    for val in data_request:
        _validate_values(val)

    return True




def api_response(data_request):
    try:
        data_request = np.asarray(data_request)
        if validate_input(data_request):
            response = predict(data_request)
            response = {"response": response}
            return response

    except Exception as e:
        response = {"response": str(e)}

    return response