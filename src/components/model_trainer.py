#important libraries
import os
import sys 

from catboost import CatBoostRegressor

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_models

from dataclasses import dataclass
@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        #inside this variable i will get train_model_file_path
        self.model_trainer_config = ModelTrainerConfig()

    #start model training
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test data input")
            #inside tuple
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            #dictionary of models, models i will be trying
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            } 

            #model score are stored model_report
            model_report: dict = evaluate_models(X_train = X_train,y_train = y_train, 
                                                 X_test = X_test, y_test = y_test, models = models)

            #gets the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))


            #nested list
            #finding which model is connected to the best score
            model_best_name = list(model_report.keys())[

                list(model_report.values()).index(best_model_score)
            ]

            #pick the name from models{}
            best_model = models[model_best_name]

            if best_model_score < 0.6:
                raise CustomException("all model scores less that 60%, no best model found")

            logging.info(f"best model found on the training and testing dataset")


            save_obj(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            final_r2 = r2_score(y_test, predicted)
            return final_r2

        except Exception as e:
            raise CustomException(e, sys)

