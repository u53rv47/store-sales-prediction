import os
import pymongo

print('IMAGE_NAME:', os.getenv("IMAGE_NAME"))
print('AWS_ACCESS_KEY_ID:', os.getenv("AWS_ACCESS_KEY_ID"))
print('AWS_SECRET_ACCESS_KEY:', os.getenv("AWS_SECRET_ACCESS_KEY"))
print('AWS_DEFAULT_REGION:', os.getenv("AWS_DEFAULT_REGION"))
print('BUCKET_NAME:', os.getenv("BUCKET_NAME"))

mongo_db_url = os.getenv("MONGO_DB_URL")
print('MONGO_DB_URL:', mongo_db_url)
mongo_client = pymongo.MongoClient(mongo_db_url)
TARGET_COLUMN = "Item_Outlet_Sales"
