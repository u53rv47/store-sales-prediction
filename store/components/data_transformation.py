from store.entity import artifact_entity, config_entity
from store.exception import StoreException
from store.logger import logging
from typing import Optional
import os
import sys
from sklearn.pipeline import Pipeline
import pandas as pd
from store import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from store.config import TARGET_COLUMN


def data_label(df: pd.DataFrame) -> pd.DataFrame:

    df_no_null = df[df.Item_Weight.isnull(
    ) == False]
    df_null = df[df.Item_Weight.isnull() == True]

    item_avg = df_no_null[['Item_Identifier', 'Item_Weight']].groupby(
        by='Item_Identifier', as_index=False).mean()

    tmp_data = pd.merge(right=df_null.drop('Item_Weight', axis=1), left=item_avg,
                        right_on='Item_Identifier', left_on='Item_Identifier', how='inner')

    df = pd.concat([df_no_null, tmp_data], axis=0)

    df_no_null = df[df.Outlet_Size.isnull(
    ) == False]

    df_null = df[df.Outlet_Size.isnull() == True]

    train10 = df_null[df_null['Outlet_Identifier'] == 'OUT010']
    train10.replace(np.nan, 'Medium', inplace=True)

    train17 = df_null[df_null['Outlet_Identifier'] == 'OUT017']
    train17.replace(np.nan, 'Medium', inplace=True)

    train45 = df_null[df_null['Outlet_Identifier'] == 'OUT045']
    train45.replace(np.nan, 'Small', inplace=True)

    df = pd.concat(
        [df_no_null, train10, train17, train45], axis=0)

    # Label encoding the train data
    df['Item_Fat_Content'].replace(
        {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)
    df['Outlet_Location_Type'].replace(
        {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}, inplace=True)
    df['Outlet_Type'].replace(
        {'Supermarket Type1': 0, 'Supermarket Type2': 1,
         'Supermarket Type3': 2, 'Grocery Store': 3}, inplace=True)
    df['Outlet_Size'].replace(
        {'Small': 0, 'Medium': 1, 'High': 2}, inplace=True)

    label_encoder = LabelEncoder()
    df['Item_Fat_Content'] = label_encoder.fit_transform(
        df['Item_Fat_Content'])
    df['Item_Type'] = label_encoder.fit_transform(
        df['Item_Type'])

    df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, inplace=True)
    return df


class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*10} Data Transformation {'<<'*10}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise StoreException(e, sys)

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            train_df = pd.read_csv(
                self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Handeling Missing Values and Labeling the train and test dataframes
            logging.info(
                "Handeling Missing Values and Labeling the train and test dataframes")
            train_df = data_label(train_df)
            test_df = data_label(test_df)

            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(
                TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(
                TARGET_COLUMN, axis=1)
            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # transforming input features
            scaler = StandardScaler()
            scaler.fit(input_feature_train_df)

            input_feature_train_arr = scaler.transform(input_feature_train_df)
            input_feature_test_arr = scaler.transform(input_feature_test_df)

            target_feature_train_arr = np.array(target_feature_train_df)
            target_feature_test_arr = np.array(target_feature_test_df)

            # target encoder
            train_arr = np.c_[input_feature_train_arr,
                              target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            # save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)
            utils.save_object(file_path=self.data_transformation_config.data_lable_object_path,
                              obj=data_label)
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=scaler)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                data_lable_object_path=self.data_transformation_config.data_lable_object_path,
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path
            )

            logging.info(
                f"Data transformation object {data_transformation_artifact}")

            return data_transformation_artifact

        except Exception as e:
            raise StoreException(e, sys)
