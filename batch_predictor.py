from store.pipeline.training_pipeline import start_training_pipeline
from store.pipeline.batch_prediction import start_batch_prediction

file_path = "bigmart-sales-data/test_AbJTz2l.csv"
if __name__ == "__main__":
    try:
        start_training_pipeline()
        predictions = start_batch_prediction(input_file_path=file_path)
        print(predictions)
    except Exception as e:
        print(e)
