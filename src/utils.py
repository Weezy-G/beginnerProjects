import os
import sys

import numpy as np
import pandas as pd
import dill 
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        #wb means write in binary
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

#each model uses the training data to guess the test data


#has report for models
def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:
        report = dict()

        #loop through each model
        for i in range(len(list(models))):
            model = list(models.values())[i]

            #train model
            model.fit(X_train, y_train)

            #prediction on X_train and X_test
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #use r2_score to see how well each models guesses are for the test
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            #saves the models score in report = dict()
            report[list(models.keys())[i]] = test_model_score

        return report







    except Exception as e:
        raise CustomException(e, sys)

        