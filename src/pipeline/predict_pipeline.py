import sys
import pandas as pd
import os

from src.exception import CustomException

#used to load pkl file
from src.utils import load_obj


class PredictPipeline:
    def __init__(self):
        pass
    
    #similar to model prediction pipeline
    #whats actually doing the prediction
    def predict(self, features):

        try:
            model_path = os.path.join("artifact","model.pkl")
            
            #responsible for handling the categorical features, feature scaling, etc
            preprocessor_path = os.path.join('artifact','preprocessor.pkl')

            #import and load pkl
            model = load_obj(file_path = model_path)
            preprocessor = load_obj(file_path = preprocessor_path)
            
            #scale data
            data_scaled = preprocessor.transform(features)

            #model does predictions
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
        

#responsible for mapping inputs from html to the backend of particular values
class CustomData:
    #features from data (StudentsPerformance.csv)
    def __init__(self,
        gender: str,
        race_ethnicity: str,                
        parental_level_of_education: str,     
        lunch: str,                          
        test_preparation_course: str,       
        reading_score: int,                  
        writing_score: int
        ):
        #values are coming from web application
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    #returns all input as a dataframe
    def get_data_as_frame(self):
        
        #from web application, whatever inputs will get mapped to the particular value
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
        

