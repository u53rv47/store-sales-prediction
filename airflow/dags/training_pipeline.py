import pendulum
import os
from airflow import DAG
from airflow.operators.python import PythonOperator


with DAG(
    'store_sales_training',
    default_args={'retries': 2},
    # [END default_args]
    description='Training Pipeline',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2023, 1, 25, tz="UTC"),
    catchup=False,
    tags=['example'],
) as dag:

    def training(**kwargs):
        from store.pipeline.training_pipeline import start_training_pipeline
        start_training_pipeline()

    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        aws_region = os.getenv("AWS_DEFAULT_REGION")
        print('Syncing files to S3 Bucket')
        os.system(
            f"aws s3 sync /app/artifact s3://{bucket_name}/artifacts --region {aws_region}")
        os.system(
            f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models --region {aws_region}")
        os.system(
            f"aws s3 sync /app/logs s3://{bucket_name}/logs --region {aws_region}")

    training_pipeline = PythonOperator(
        task_id="train_pipeline",
        python_callable=training

    )

    sync_data_to_s3 = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifact_to_s3_bucket

    )

    training_pipeline >> sync_data_to_s3
