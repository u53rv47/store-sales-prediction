from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator


with DAG(
    'store_sales_training',
    default_args={'retries': 2},
    # [END default_args]
    description='Batch Prediction',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2023, 1, 20, tz="UTC"),
    catchup=False,
    tags=['example'],
) as dag:

    def download_files(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        input_dir = "/app/input_files"
        # creating directory
        os.makedirs(input_dir, exist_ok=True)
        os.system(
            f"aws s3 sync s3://{bucket_name}/input_files /app/input_files")

    def batch_prediction(**kwargs):
        from store.pipeline.batch_prediction import start_batch_prediction
        input_dir = "/app/input_files"
        for file_name in os.listdir(input_dir):
            # make prediction
            start_batch_prediction(
                input_file_path=os.path.join(input_dir, file_name))

    def sync_prediction_dir_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(
            f"aws s3 sync /app/prediction s3://{bucket_name}/prediction_files")

    download_input_files = PythonOperator(
        task_id="download_file",
        python_callable=download_files

    )

    generate_prediction_files = PythonOperator(
        task_id="batch_prediction",
        python_callable=batch_prediction

    )

    upload_prediction_files = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_prediction_dir_to_s3_bucket

    )

    download_input_files >> generate_prediction_files >> upload_prediction_files
