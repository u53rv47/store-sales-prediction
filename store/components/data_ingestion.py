from store import utils
from store.entity import config_entity
from store.entity import artifact_entity
from store.exception import StoreException
from store.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*10} Data Ingestion {'<<'*10}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise StoreException(e, sys)

    def inititate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(F"Exporting Collection data as Pandas Dataframe")
            df: pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, collection_name=self.data_ingestion_config.collection_name)
            logging.info("Save  data in feature store")

            # replace na with NAN

            df.replace(to_replace="na", value=np.NAN, inplace=True)

            # create feature store folder
            feature_store_dir = os.path.dirname(
                self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info("save df to feature store folder")
            # save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,
                      index=False, header=True)

            logging.info("split dataset into train and test set")
            train_df, test_df = train_test_split(
                df, test_size=self.data_ingestion_config.test_size, random_state=42)

            logging.info("Creating dataset directory folder if not available")

            dataset_dir = os.path.dirname(
                self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info("save df to feature store folder")

            train_df.to_csv(
                path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(
                path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # Prepare artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path, train_file_path=self.data_ingestion_config.train_file_path, test_file_path=self.data_ingestion_config.test_file_path)
            logging.info(f"Data ingestion artifact:{data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise StoreException(e, sys)
