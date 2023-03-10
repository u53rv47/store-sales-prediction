import sys
from store.exception import StoreException
from store.entity import config_entity
from store.components.data_ingestion import DataIngestion
from store.components.data_validation import DataValidation
from store.components.data_transformation import DataTransformation
from store.components.model_trainer import ModelTrainer
from store.components.model_evaluation import ModelEvaluation
from store.components.model_pusher import ModelPusher
from store.logger import LOG_FILE_PATH, logging
from store.utils import save_object


def start_training_pipeline():
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

        data_tranformation_config = config_entity.DataTransformationConfig(
            training_pipeline_config=training_pipeline_config)
        data_tranformation = DataTransformation(
            data_transformation_config=data_tranformation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_tranformation.initiate_data_transformation()

        model_trainer_config = config_entity.ModelTrainerConfig(
            training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        model_eval_config = config_entity.ModelEvaluationConfig(
            training_pipeline_config=training_pipeline_config)
        model_eval = ModelEvaluation(model_eval_config=model_eval_config, data_ingestion_artifact=data_ingestion_artifact,
                                     data_transformation_artifact=data_transformation_artifact, model_trainer_artifact=model_trainer_artifact)
        model_eval_artifact = model_eval.initiate_model_evaluation()

        model_pusher_config = config_entity.ModelPusherConfig(
            training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
                                   data_transformation_artifact=data_transformation_artifact, model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()

        save_object(file_path=LOG_FILE_PATH, obj=logging)

    except Exception as e:
        raise StoreException(e, sys)
