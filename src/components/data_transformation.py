from src.entity import artifact_entity,config_entity
from src.exception import CustomException
from src.logger import logging
from typing import Optional
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from src import utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from src.config import TARGET_COLUMN,num_col,cat_col
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)


    @classmethod
    def get_data_transformer_object_input(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0,add_indicator=True)
            robust_scaler =  RobustScaler()
            # ohe_enc = make_column_transformer((OneHotEncoder(sparse_output=False),cat_col), remainder='passthrough')
            # pipeline_input = make_pipeline(ohe_enc,simple_imputer,robust_scaler)
            pipeline_input = make_pipeline(simple_imputer,robust_scaler)
            return pipeline_input
        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def get_data_transformer_object_target(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0,add_indicator=True)
            robust_scaler =  RobustScaler()
            pipeline_target = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ('RobustScaler',robust_scaler)
                ])
            return pipeline_target
        except Exception as e:
            raise CustomException(e, sys)

    # @classmethod    
    def detect_outliers_iqr(data):
        outliers=[]
        data = sorted(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        IQR = q3-q1
        lwr_bound = q1-(1.5*IQR)
        upr_bound = q3+(1.5*IQR)
        for i in data:
            if (i<lwr_bound or i>upr_bound):
                outliers.append(i)
        return outliers


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            #selecting input feature for train and test dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN+cat_col,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN+cat_col,axis=1)
            # print(train_df.columns)

            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            #removing outliers
            for i in range(-3,0):
                data = sorted(train_df.iloc[:,i])
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                # print(q1, q3)
                IQR = q3-q1
                lwr_bound = q1-(1.5*IQR)
                upr_bound = q3+(1.5*IQR)
                for j in range(len(train_df.iloc[:,i])):
                    if train_df.iloc[j,i] in DataTransformation.detect_outliers_iqr(data):
                        train_df.iloc[j,i]=np.where(train_df.iloc[j,i]>upr_bound,upr_bound,np.where(train_df.iloc[j,i]<lwr_bound,lwr_bound,train_df.iloc[j,i]))

            #fit train input features
            transformation_pipeline = DataTransformation.get_data_transformer_object_input()
            transformation_pipeline.fit(input_feature_train_df)

            #transforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            transformation_pipeline_target = DataTransformation.get_data_transformer_object_target()
            transformation_pipeline_target.fit(target_feature_train_df)

            target_feature_train_arr = transformation_pipeline_target.transform(target_feature_train_df)
            target_feature_test_arr =transformation_pipeline_target.transform(target_feature_test_df)
            

            #smt = SMOTETomek(random_state=42)
            #logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            #input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            #logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            #logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            #input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            #logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipeline)

            utils.save_object(file_path=self.data_transformation_config.transformed_target_path,
             obj=transformation_pipeline_target)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                transformed_target_path = self.data_transformation_config.transformed_target_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)