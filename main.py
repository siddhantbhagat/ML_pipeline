from src.pipeline.training_pipeline import start_training_pipeline
from src.pipeline.prediction_pipeline import start_batch_prediction

file_path="notebooks\MatNavi Mechanical properties of low-alloy steels.csv"
print(__name__)
if __name__=="__main__":
     try:
          start_training_pipeline()
          output_file = start_batch_prediction(input_file_path=file_path)
          print(output_file)
     except Exception as e:
          print(e)