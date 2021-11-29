import argparse
# import sys
# sys.path.append("..")
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from src import load_data
from src import split_data
from src import train_and_evaluate
from src import log_production_model

default_args ={
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021,11,24),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
    }


dag = DAG(
    'airflow_cps_dag',
    default_args=default_args,
    description='Dag for predictive maintenance',
    schedule_interval= timedelta(days=7),
    catchup = False
    )

def execute_mlops_pipeline():

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="/usr/local/airflow/params.yaml")
    parsed_args = args.parse_args()
    load_data.load_and_save(config_path=parsed_args.config)
    split_data.split_and_saved_data(config_path=parsed_args.config)
    train_and_evaluate.train_and_evaluate(config_path=parsed_args.config)
    log_production_model.log_production_model(config_path=parsed_args.config)
    print('done')

run_etl = PythonOperator(
    task_id='cps_pipeline',
    python_callable=execute_mlops_pipeline,
    dag=dag,
)
run_etl



