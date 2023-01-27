import os
import sys
import numpy as np
from store.exception import StoreException
from store.logger import logging
from store.predictor import ModelResolver
import pandas as pd
from store.utils import load_object
from datetime import datetime
from store.logger import LOG_FILE_PATH, logging
from store.utils import save_object
PREDICTION_DIR = "prediction"


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na": np.NAN}, inplace=True)
        # validation

        logging.info(f"Loading Data Label to label the dataset")
        data_label = load_object(
            file_path=model_resolver.get_latest_data_lable_path())
        df = data_label(df)

        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(
            file_path=model_resolver.get_latest_transformer_path())

        input_feature_names = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        df["predicted_sales"] = prediction

        prediction_file_name = os.path.basename(input_file_path).replace(
            ".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(
            PREDICTION_DIR, prediction_file_name)
        logging.info(f'Prediction file is saved at "{prediction_file_path}"')
        df.to_csv(prediction_file_path, index=False, header=True)

        save_object(file_path=LOG_FILE_PATH, obj=logging)
        return prediction_file_path
    except Exception as e:
        raise StoreException(e, sys)
