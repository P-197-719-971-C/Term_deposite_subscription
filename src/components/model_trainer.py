import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint, uniform


from src.exception.exception import CustomException
from src.logging.logger import logging

from src.utils.common import save_object,evaluate_classification_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, df_arr, train_array, valid_array, test_array, df_arr_encoded, train_arr_encoded, valid_arr_encoded, test_arr_encoded):
        try:
            logging.info("Splitting training, valid and test data")

            X, y, X_train, y_train, X_valid, y_valid, X_test, y_test = (
                df_arr[:, :-1],
                df_arr[:, -1],
                train_array[:,:-1],
                train_array[:,-1],
                valid_array[:,:-1],
                valid_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            X_encoded, y, X_train_encoded, y_train, X_valid_encoded, y_valid, X_test_encoded, y_test = (
                df_arr_encoded[:, :-1],
                df_arr_encoded[:, -1],
                train_arr_encoded[:,:-1],
                train_arr_encoded[:,-1],
                valid_arr_encoded[:,:-1],
                valid_arr_encoded[:,-1],
                test_arr_encoded[:,:-1],
                test_arr_encoded[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Support Vector Classifier": SVC(),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "XGBoost Classifier": XGBClassifier(scale_pos_weight=float(np.sum(y_train == 0)) /(3*np.sum(y_train == 1)))
                
            }
            """
            "CatBoost Classifier": CatBoostClassifier(verbose=False)
            """

            
            params={
            "Logistic Regression": {
                'solver' :["saga", "liblinear"],
                'penalty':['l1', 'l2'],
                'C':[0.01, 0.03, 0.1, 1, 10],
                'max_iter' : [1000]
            },
            "K-Neighbors Classifier": {
                'n_neighbors': np.arange(13,50).tolist()
            },
            "Support Vector Classifier": {
                'C': [0.1, 1],
                'kernel': ['linear', 'rbf'],
                'gamma': [0.1, 1]
            },
            "Naive Bayes":{
                'var_smoothing': np.logspace(-9, 0, 100)
            },
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                'splitter':['best','random'],
                'max_features':['sqrt','log2'],
            },
            "Random Forest": {
                'n_estimators': [100,128,256,300,400],
                'criterion':['gini'],
                'max_features':['sqrt','log2'],
                'max_depth': [None, 10, 20],  # Limit the maximum depth of trees
                'min_samples_split': [2, 5, 10],  # Adjust minimum samples split
                'min_samples_leaf': [1, 2, 4] 
            },
            "AdaBoost Classifier": {
                'n_estimators': [64,128,256],
                'learning_rate': [0.001, 0.01, 0.1]
            },
            "XGBoost Classifier": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'max_depth': randint(3, 19),  # Random integer between 3 and 18
                'gamma': uniform(1, 9),  # Random float between 1 and 10
                'reg_alpha': randint(40, 181),  # Random integer between 40 and 180
                'reg_lambda': uniform(0, 1),  # Random float between 0 and 1
                'colsample_bytree': uniform(0.5, 0.5),  # Random float between 0.5 and 1
                'min_child_weight': randint(0, 11)  # Random integer between 0 and 10
            }
            }
            ''',
            "CatBoost Classifier": {
                'iterations': np.arange(100, 1300, 100).tolist(),
                'learning_rate': [0.001, 0.004, 0.01, 0.1, 0.2, 0.3],
                'depth': np.arange(1, 11).tolist(),
                'l2_leaf_reg': [1, 3, 5, 7, 9],
                'boosting_type': ['Ordered', 'Plain']
            }'''
            
            scores = ['recall']

            logging.info("Evaluating best model")
               

            results_df, classifier = evaluate_classification_model(X, y, X_train, y_train, X_valid, y_valid, X_test, y_test, X_encoded, X_train_encoded, X_valid_encoded , X_test_encoded, models, params, cv=3, n_iter=5)
            


            
            logging.info("Checking if best model has recall is more than 0.6 and precision is greater than 0.5")

            recall_df = results_df[(results_df["Test Recall"] > 0.6) & (results_df["Test Precision"] > 0.5)][["Model","Test Recall", "Test Precision"]]

            if len(recall_df) == 0:
                raise CustomException("No best model found")
            else:
                logging.info("Extracting best model name from recall")

                best_model_name = recall_df.loc[recall_df['Test Recall'].idxmax(), "Model"]
                best_model = classifier[best_model_name]
                
                logging.info("Best model found on training,  Valid and testing datasets")

                logging.info("Saving best model")

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj = best_model
                
                )
                
            
            return recall_df, best_model_name, best_model, results_df, classifier 

        except Exception as e:
            raise CustomException(e, sys)
