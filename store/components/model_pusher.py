from store.predictor import ModelResolver
from store.entity.config_entity import ModelPusherConfig
from store.exception import StoreException
import os
import sys
from store.utils import load_object, save_object
from store.logger import logging
from store.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*10} Model Pusher {'<<'*10}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(
                model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise StoreException(e, sys)

    def initiate_model_pusher(self,) -> ModelPusherArtifact:
        try:
            # load object
            logging.info(f"Loading data_lable, transformer and model")
            data_lable = load_object(
                file_path=self.data_transformation_artifact.data_lable_object_path)
            transformer = load_object(
                file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(
                file_path=self.model_trainer_artifact.model_path)

            # model pusher dir
            logging.info(f"Saving model into model pusher directory")
            save_object(
                file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(
                file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(
                file_path=self.model_pusher_config.pusher_data_lable_path, obj=data_lable)

            # saved model dir
            logging.info(f"Saving model in saved model dir")
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            data_lable_path = self.model_resolver.get_latest_save_data_lable_path()

            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
            save_object(file_path=data_lable_path, obj=data_lable)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise StoreException(e, sys)
