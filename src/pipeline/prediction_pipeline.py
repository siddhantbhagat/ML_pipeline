from src.exception import CustomException
from src.logger import logging
from src.predictor import ModelResolver
import pandas as pd
from src.utils import load_object
import os,sys
from datetime import datetime
PREDICTION_DIR="prediction"

import numpy as np
def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        #data ingestion
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na":np.NAN},inplace=True)
        #data validation
        #data transformation
        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        
        input_feature_names =  list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])
        #prediction
        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)
        prediction = prediction.reshape(-1,1)
        logging.info(f"Target encoder to convert predicted column")
        target_transformer = load_object(file_path=model_resolver.get_latest_target_transformer_path())

        APP_prediction = target_transformer.inverse_transform(prediction)

        df["prediction"]=prediction
        df["APP_prediction"]=APP_prediction


        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path
    except Exception as e:
        raise CustomException(e, sys)