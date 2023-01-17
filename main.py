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


# file_path = "aps_failure_training_set1.csv"
if __name__ == "__main__":
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(
            training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(
            data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.inititate_data_ingestion()

        data_validation_config = config_entity.DataValidationConfig(
            training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(
            data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        # start_training_pipeline()
        # output_file = start_batch_prediction(input_file_path=file_path)
        # print(output_file)
    except Exception as e:
        print(e)
