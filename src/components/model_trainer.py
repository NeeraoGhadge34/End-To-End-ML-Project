import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor)
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting data into train and test input.")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "K Neighbors Regression" : KNeighborsRegressor(),
                "Random Forest Regression" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "ADA-Boost Regression" : AdaBoostRegressor(),
                "XG Boost Regression" : XGBRegressor(),
                "Cat Boost Regression" : CatBoostRegressor(verbose = False)
            }

            params = {
                "Linear Regression" : {},

                "K Neighbors Regression" : {
                    "n_neighbors": [3,5,7,9],
                    "weights": ["uniform","distance"],
                    "algorithm": ["auto","ball_tree","kd_tree","brute"],
                    "p": [1,2]
                    },

                "Random Forest Regression" : {
                    "n_estimators": [100,200,500],
                    "max_depth": [None,10,20,30],
                    "min_samples_split": [2,5,10],
                    #"min_samples_leaf": [1,2,4],
                    #"max_features": ["sqrt","log2"]
                    },
                
                "Decision Tree" : {
                    "criterion": ["squared_error","friedman_mse"],
                    "splitter": ["best","random"],
                    "max_depth": [None,10,20,30],
                    #"min_samples_split": [2,5,10],
                    #"min_samples_leaf": [1,2,4]
                    },

                "ADA-Boost Regression" : {
                    "n_estimators": [50,100,200],
                    "learning_rate": [0.01,0.1,0.001],
                    "loss": ["linear","square","exponential"]
                    },

                "XG Boost Regression" : {
                    "n_estimators": [100,200],
                    "learning_rate": [0.01,0.1,0.2],
                    "max_depth": [3,5,7],
                    #"subsample": [0.8,1],
                    #"colsample_bytree": [0.8,1]
                    },

                "Cat Boost Regression" : {
                    "iterations": [100,200],
                    "learning_rate": [0.01,0.1],
                    "depth": [4,6,8],
                    #"l2_leaf_reg": [1,3,5]
                }
            }

            model_report: dict =  evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            # Score of best model
            best_model_score = max(sorted(model_report.values()))

            # Name of best model
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.")
            logging.info("Best model found for training and testing data.")

            save_object(file_path = self.model_trainer_config.trained_model_file_path,
                        obj = best_model)

            prediction = best_model.predict(X_test)
            r2_squared = r2_score(y_test, prediction)

            return r2_squared
        
        except Exception as e:
            raise CustomException(e,sys)
