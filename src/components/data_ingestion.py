import os
import sys

#calling custom exception
from src.exception import CustomException

#able to log data ingestion
from src.logger import logging

import pandas as pd

#used for train test split
from sklearn.model_selection import train_test_split

#used to create class variables
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
#anything i need i will give to this
#inputs im giving to data ingestion components
#knows where to save
class dataIngestionConfig:
    #define class variable
    #outputs saved in artifact
    trian_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "raw.csv")

class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("enterd the data ingestion method")
        try:
            #read the dataset (can be from anywhere)
            df = pd.read_csv("notebook/StudentsPerformance.csv")
            logging.info("read data as dataframe")
            
            #if its already there, keep the particular folder
            os.makedirs(os.path.dirname(self.ingestion_config.trian_data_path), exist_ok = True)

            #save data in this path
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info("train test split initiated")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

            train_set.to_csv(self.ingestion_config.trian_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("ingestion of data complete")

            #returing train and test data path
            return(
                self.ingestion_config.trian_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

#initiate and run
if __name__ == "__main__":

    #combined data ingestion, then combined data transformation
    obj = dataIngestion()

    #returning these 2 values
    train_data,test_data = obj.initiate_data_ingestion()



    #calls the DataTransformation() class in data_transformation.py
    data_transformation = DataTransformation()

    #calls the initiate_data_transformation function in data_transformation.py
    train_arrary, test_array,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arrary, test_array))