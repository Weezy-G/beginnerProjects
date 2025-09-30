#feature engineering + data cleaning + coonverting

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

#ColumnTransformer used to create pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_obj

@dataclass
#any path/inputs i may need
class DataTransformationConfig:
    #get artifact folder
    #creating pipeline pkl file named = preprocessor.pkl
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")


#the input i will be giving
class DataTransformation:
    def __init__(self):
        #will have this variable "preprocessor_obj_file_path"
        self.data_transformation_config = DataTransformationConfig()

    #create all pkl file
    #responsible for converting categorical features into numerical for standard scalars
    def get_data_transformer_obj(self):
        '''
        responsible for data transformation based on  different types of data
        '''


        try:
            numeric_features = ["writing score", "reading score"]
            categorical_features = [
                "gender", 
                "race/ethnicity", 
                "parental level of education", 
                "lunch", 
                "test preparation course"
            ]
            #things to do:
            #1. create pipeline
            #2. handle missing values

            num_pipeline = Pipeline(

                steps = [
                    #imputer is respondible for handling missing values
                    ("imputer", SimpleImputer(strategy = "median")),
                    #handle standard Scaler
                    ("Scaler", StandardScaler())
                ]
            )

            #handling missing values in categorical and converting to a numerical values
            cat_pipeline = Pipeline(

                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"numerical column standard scaler done {categorical_features}")
            logging.info(f"categorical column encoding done {numeric_features}")
 
            #combining numerical and categorical together
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, numeric_features),
                 ("cat_pipeline", cat_pipeline, categorical_features)
                 ]

                )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    #data transformation techniques
    #starting data transformation within this function

    #train_path and test_path come from data ingestion path
    def initiate_data_transformation(self, trian_path, test_path):

        try:
            #read in datasets
            train_df = pd.read_csv(trian_path)
            test_df = pd.read_csv(test_path)

            #log details
            logging.info("train and test has been read")

            logging.info("getting preprocessing object")

            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = "math score"
            numeric_features = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"applying precossing objects on training and test dataframe")

            input_feature_train_df_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_array = preprocessor_obj.transform(input_feature_test_df)

            #concatenate the target feature to the end of the input features
            train_array = np.c_[
                input_feature_train_df_array, np.array(target_feature_train_df)
            ]

            test_array = np.c_[
                input_feature_test_df_array, np.array(target_feature_test_df)
            ]


            logging.info(f"saved processed objects")


            #used to save pkl file
            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)