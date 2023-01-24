from store.pipeline.training_pipeline import start_training_pipeline
from store.pipeline.batch_prediction import start_batch_prediction


from store.logger import logging
from store.exception import StoreException
from store.utils import get_collection_as_dataframe
import sys
import os
from store.entity import config_entity
from store.components.data_ingestion import DataIngestion
from store.components.data_validation import DataValidation
from store.components.data_transformation import DataTransformation
from store.components.model_trainer import ModelTrainer
from store.components.model_evaluation import ModelEvaluation
from store.components.model_pusher import ModelPusher


file_path = "bigmart-sales-data/test_AbJTz2l.csv"
if __name__ == "__main__":
    try:

        # start_training_pipeline()
        predictions = start_batch_prediction(input_file_path=file_path)
        # print(predictions)
    except Exception as e:
        print(e)
