to execute code run these commands in order


pip install -r requirements.txt

dvc repro


mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0 -p 5001


airflow standalone