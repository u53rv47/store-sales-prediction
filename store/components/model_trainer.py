from store.entity import artifact_entity, config_entity
from store.exception import StoreException
from store.logger import logging
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from store import utils
from sklearn.metrics import r2_score, mean_squared_error


class ModelTrainer:

    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig, data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*10} Model Trainer {'<<'*10}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise StoreException(e, sys)

    def train_model(self, x, y):
        try:
            best_param = {'criterion': 'poisson',
                          'max_depth': 6,
                          'max_features': 7,
                          'min_samples_leaf': 2,
                          }
            rfr = RandomForestRegressor(**best_param)
            rfr.fit(x, y)
            return rfr
        except Exception as e:
            raise StoreException(e, sys)

    def initiate_model_trainer(self,) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(
                f"Splitting input and target feature from both train and test arr.")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Train the model")
            model = self.train_model(x=x_train, y=y_train)

            logging.info(f"Calculating adj_r2 train score")
            yhat_train = model.predict(x_train)

            rmse_train = np.sqrt(mean_squared_error(
                y_true=y_train, y_pred=yhat_train))
            r2_train = r2_score(y_true=y_train, y_pred=yhat_train)
            adj_r2_train = adj_r2 = 1 - \
                (1-r2_train)*(len(y_train)-1) / \
                (len(y_train)-x_train.shape[1]-1)

            logging.info(f"Calculating adj_r2 test score")
            yhat_test = model.predict(x_test)

            rmse_test = np.sqrt(mean_squared_error(
                y_true=y_test, y_pred=yhat_test))
            r2_test = r2_score(y_true=y_test, y_pred=yhat_test)
            adj_r2_test = 1 - (1-r2_test)*(len(y_test)-1) / \
                (len(y_test)-x_test.shape[1]-1)

            logging.info(
                f"rmse train: {rmse_train} and rmse_test: {rmse_test}")
            logging.info(
                f"r2_train: {r2_train} and r2_test: {r2_test}")
            logging.info(
                f"adj_r2_train: {adj_r2_train} and adj_r2_test: {adj_r2_test}")
            # check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            # check for overfitting or underfiiting or expected score
            if adj_r2_test < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {adj_r2_test}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(adj_r2_train-adj_r2_test)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(
                    f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            # save the trained model
            logging.info(f"Saving model object")
            utils.save_object(
                file_path=self.model_trainer_config.model_path, obj=model)

            # prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path, adj_r2_train=adj_r2_train, adj_r2_test=adj_r2_test)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise StoreException(e, sys)
