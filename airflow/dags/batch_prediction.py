from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator


with DAG(
    'Store_sales_training',
    default_args={'retries': 2},
    # [END default_args]
    description='Store Sales Prediction',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2023, 1, 23, tz="UTC"),
    catchup=False,
    tags=['example'],
) as dag:

    def training(**kwargs):
        from store.pipeline.batch_prediction import start_batch_prediction
        start_batch_prediction()

    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync /app/artifact s3://{bucket_name}/artifacts")
        os.system(
            f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models")

    training_pipeline = PythonOperator(
        task_id="batch_prediction",
        python_callable=training

    )

    sync_data_to_s3 = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifact_to_s3_bucket

    )

    training_pipeline >> sync_data_to_s3
