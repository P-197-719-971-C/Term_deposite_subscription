import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier)


from src/BankMarketing.exception.exception import CustomException
from src/BankMarketing.logging.logger import logging

from src/BankMarketing.utils.common import save_object,evaluate_classification_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SVC": SVC(),
                "CatBoostClassifier": CatBoostClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
            }
            
            params={
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter':['best','random'],
                    #'max_features':['sqrt','log2'],
                },
                "Logistic Regression": {
                    'penalty':['l1', 'l2'],
                    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]
                },
                "XGBClassifier": {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsClassifier": {
                    'n_neighbors':[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
                },
                "SVC": {
                    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'kernel':['linear', 'poly', 'rbf','sigmoid']
                },
                "CatBoostClassifier": {
                    'iterations':[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,20]
                },
                "AdaBoostClassifier": {'n_estimators': [8,16,32,64,128,256],
                'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 1.0]
            }
            }
     
            logging.info("Evaluating best model")
               

            model_report: dict = evaluate_classification_model(X_train=X_train, y_train=y_train, X_test=X_test,
                                                  y_test=y_test, models = models, param=params)
            
            

            logging.info("Printing f1 scores")

            for model_name, metrics in model_report.items():
                score_f1 = metrics['f1']
                print(f"f1 score for {model_name}: {score_f1}")

            logging.info("Best model found on testing dataset")

            best_model_name = max(model_report, key=lambda model: model_report[model]['f1'])
            best_f1_score = model_report[best_model_name]['f1']
            best_model = models[best_model_name]

            logging.info("Checking if best model has f1 score more than 0.6")
            
            if best_f1_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing datasets")

            logging.info("Saving best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            
            )
            logging.info("returning best f1 score")
            
            predicted = best_model.predict(X_test)
            score_f1 = f1_score(y_test, predicted)
            return score_f1

        except Exception as e:
            raise CustomException(e, sys)
