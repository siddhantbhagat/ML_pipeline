import os
import sys
from src.logger import logging
from src.exception import CustomException
from src import utils
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.entity import artifact_entity,config_entity


# @dataclass
# class DataIngestionConfig:
#     train_file_path:str = os.path.join("artifacts","train.csv")
#     test_file_path:str = os.path.join("artifacts","test.csv")
#     feature_store_file_path:str = os.path.join("artifacts","data.csv")

class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig ):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame  = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)

            logging.info("Save data in feature store")

            #replace na with Nan
            df.replace(to_replace="na",value=np.NAN,inplace=True)

            #Save data in feature store
            logging.info("Create feature store folder if not available")
            #Create feature store folder if not available
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)


            logging.info("split dataset into train and test set")
            #split dataset into train and test set
            dataset_size = len(df)
            test_size = 0.3
            train_df,test_df = df.iloc[int(dataset_size*test_size):],df.iloc[:int(dataset_size*(test_size))]
            train_df=train_df.sample(frac=1)
            # train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)
            
            logging.info("create dataset directory folder if not available")
            #create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)
            
            #Prepare artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(error_message=e, error_detail=sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()

# class DataIngestion:
#     def __init__(self):
#         self.data_ingestion_config = DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         logging.info('Entered Data ingestion component')
#         try:
#             df = pd.read_csv('notebooks\MatNavi Mechanical properties of low-alloy steels.csv')
#             dataset_size = len(df)
#             test_size = 0.3
#             logging.info('Read data in Data frame')
#             os.makedirs(os.path.dirname(self.data_ingestion_config.train_file_path), exist_ok=True)
#             df.to_csv(self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            
#             logging.info('Train-test split initiated')
#             # Using Train test is leading to overfitting hence we manually split the data.
#             train_set,test_set = df.iloc[int(dataset_size*test_size):],df.iloc[:int(dataset_size*(test_size))]
#             train_set=train_set.sample(frac=1)

#             train_set.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
#             test_set.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)

#             logging.info('data ingestion is completed successfully')

#             #Prepare artifact

#             data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
#                 feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
#                 train_file_path=self.data_ingestion_config.train_file_path, 
#                 test_file_path=self.data_ingestion_config.test_file_path)

#             logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
#             return data_ingestion_artifact


#         except Exception as e:
#             raise CustomException(e,sys)
        
