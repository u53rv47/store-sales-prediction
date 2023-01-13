import pymongo
import pandas as pd
import json
from store.config import mongo_client


# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient()

DATABASE_NAME = "store"
COLLECTION_NAME = "sales"
DATA_FILE_PATH = "bigmart-sales-data/train_v9rqX0R.csv"


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    # convert dataframe into json so that we can dump the records into mongodb

    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
