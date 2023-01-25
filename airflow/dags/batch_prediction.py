from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator


with DAG(
    'store_sales_batch_prediction',
    default_args={'retries': 2},
    # [END default_args]
    description='Store Sales Batch Prediction',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2023, 1, 25, tz="UTC"),
    catchup=False,
    tags=['example'],
) as dag:

    def batch_prediction(**kwargs):
        from store.pipeline.batch_prediction import start_batch_prediction
        start_batch_prediction()

    def sync_prediction_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync /app/prediction s3://{bucket_name}/prediction")

    batch_prediction = PythonOperator(
        task_id="batch_prediction",
        python_callable=batch_prediction

    )

    sync_data_to_s3 = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_prediction_to_s3_bucket

    )

    batch_prediction >> sync_data_to_s3
