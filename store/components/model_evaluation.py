from store.predictor import ModelResolver
from store.entity import config_entity, artifact_entity
from store.exception import StoreException
from store.logger import logging
from store.utils import load_object
import pandas as pd
import sys
import os
import numpy as np
from store.config import TARGET_COLUMN
from sklearn.metrics import r2_score, mean_squared_error


class ModelEvaluation:

    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainerArtifact
                 ):
        try:
            logging.info(f"{'>>'*10}  Model Evaluation {'<<'*10}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise StoreException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            # if saved model folder has model the we will compare
            # which model is best trained or the model from saved model folder
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                              improved_accuracy=None)
                logging.info(
                    f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Finding location of transformer model and target encoder
            logging.info(
                "Finding location of transformer and model")
            data_lable_path = self.model_resolver.get_latest_data_lable_path()
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()

            logging.info(
                "Previous trained objects of transformer and model")
            # Previous trained  objects
            data_lable = load_object(file_path=data_lable_path)
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)

            logging.info("Currently trained model objects")
            # Currently trained model objects
            current_data_lable = load_object(
                file_path=self.data_transformation_artifact.data_lable_object_path)
            current_transformer = load_object(
                file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(
                file_path=self.model_trainer_artifact.model_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # accuracy using previous trained model
            input_df = data_lable(df=test_df)

            input_arr = transformer.transform(
                input_df.drop(TARGET_COLUMN, axis=1))
            target_df = input_df[TARGET_COLUMN]

            y_pred = model.predict(input_arr)
            y_true = np.array(target_df)

            previous_model_rmse = np.sqrt(mean_squared_error(
                y_true=y_true, y_pred=y_pred))
            previous_model_r2_score = r2_score(y_true=y_true, y_pred=y_pred)
            previous_model_adj_r2_score = 1 - \
                (1-previous_model_r2_score)*(len(y_true)-1) / \
                (len(y_true) - input_arr.shape[1]-1)
            logging.info(
                f"Accuracy using previous trained model:\nRMSE: {previous_model_rmse} \nR2 Score: {previous_model_r2_score} \nAdj R2 Score: {previous_model_adj_r2_score} ")

            # accuracy using current trained model
            input_df = current_data_lable(test_df)

            input_arr = current_transformer.transform(
                input_df.drop(TARGET_COLUMN, axis=1))
            target_df = input_df[TARGET_COLUMN]

            y_pred = current_model.predict(input_arr)
            y_true = np.array(target_df)

            current_model_rmse = np.sqrt(mean_squared_error(
                y_true=y_true, y_pred=y_pred))
            current_model_r2_score = r2_score(y_true=y_true, y_pred=y_pred)
            current_model_adj_r2_score = 1 - \
                (1-current_model_r2_score)*(len(y_true)-1) / \
                (len(y_true) - input_arr.shape[1]-1)
            logging.info(
                f"Accuracy using current trained model:\nRMSE: {current_model_rmse} \nR2 Score: {current_model_r2_score} \nAdj R2 Score: {current_model_adj_r2_score} ")

            if current_model_rmse <= previous_model_rmse:
                logging.info(
                    f"Current trained model is better than previous model")
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True, improved_accuracy=current_model_r2_score-previous_model_r2_score)
                logging.info(f"Model eval artifact: {model_eval_artifact}")
                return model_eval_artifact
            raise Exception(
                "Current trained model is not better than previous model")
        except Exception as e:
            raise StoreException(e, sys)
