import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.logging.logger import logging

from src.exception.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
              
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
        
def evaluate_classification_model(X, y, X_train, y_train, X_valid, y_valid, X_test, y_test, X_encoded, X_train_encoded, X_valid_encoded , X_test_encoded, models, params, cv=3, n_iter=5):
    try:    
        results = []  # List to store dictionaries of results and their metrics
        classifier = {}
        for model_name, model in models.items():
            param = params[model_name]     
            logging.info(f"Model {model_name} has started training")

            if model_name == 'CatBoost Classifier':
                rs = RandomizedSearchCV(model, param_distributions=param, scoring="recall", cv=cv, n_iter=n_iter)
                rs.fit(X_encoded, y)
                best_params = rs.best_params_
                classifier[model_name] = model

                classifier[model_name].set_params(**best_params)

                classifier[model_name].fit(X_train_encoded, y_train)

                y_train_pred = classifier[model_name].predict(X_train_encoded)
                y_valid_pred = classifier[model_name].predict(X_valid_encoded)
                y_test_pred = classifier[model_name].predict(X_test_encoded)

            else:
                
                rs = RandomizedSearchCV(model, param_distributions=param, scoring="recall", cv=cv, n_iter=n_iter)
                rs.fit(X, y)
                best_params = rs.best_params_
                classifier[model_name] = model

                classifier[model_name].set_params(**best_params)

                classifier[model_name].fit(X_train, y_train)

                y_train_pred = classifier[model_name].predict(X_train)
                y_valid_pred = classifier[model_name].predict(X_valid)
                y_test_pred = classifier[model_name].predict(X_test)

            accuracy_train = round(accuracy_score(y_train, y_train_pred), 2)
            accuracy_valid = round(accuracy_score(y_valid, y_valid_pred), 2)
            accuracy_test = round(accuracy_score(y_test, y_test_pred), 2)

            precision_train = round(precision_score(y_train, y_train_pred), 2)
            precision_valid = round(precision_score(y_valid, y_valid_pred), 2)
            precision_test = round(precision_score(y_test, y_test_pred), 2)

            recall_train = round(recall_score(y_train, y_train_pred), 2)
            recall_valid = round(recall_score(y_valid, y_valid_pred), 2)
            recall_test = round(recall_score(y_test, y_test_pred), 2)

            f1_train = round(f1_score(y_train, y_train_pred), 2)
            f1_valid = round(f1_score(y_valid, y_valid_pred), 2)
            f1_test = round(f1_score(y_test, y_test_pred), 2)

            

            

            model_metrics = {
                'Model': model_name,
                'Metrics': {
                    'Train': {'Accuracy': accuracy_train, 'Precision': precision_train, 'Recall': recall_train, 'F1 Score': f1_train},
                    'Validation': {'Accuracy': accuracy_valid, 'Precision': precision_valid, 'Recall': recall_valid, 'F1 Score': f1_valid},
                    'Test': {'Accuracy': accuracy_test, 'Precision': precision_test, 'Recall': recall_test, 'F1 Score': f1_test},
                },
                'Best Parameters': best_params
            }

            # Append the model metrics dictionary to the list of classifiers
            results.append(model_metrics)
            # Create a DataFrame from the list of classifiers
            results_df = pd.DataFrame(results)

            # Extract metrics from the nested 'Metrics' dictionary and flatten the structure
            metrics_df = pd.concat([results_df['Metrics'].apply(lambda x: x[key]).apply(pd.Series).add_prefix(key + ' ') for key in ['Train', 'Validation', 'Test']], axis=1)

            # Concatenate extracted metrics with the original DataFrame and rearrange columns
            results_df = pd.concat([results_df.drop(['Metrics'], axis=1), metrics_df], axis=1)[
                ['Model', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1 Score',
                'Validation Accuracy', 'Validation Precision', 'Validation Recall', 'Validation F1 Score',
                'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score', 'Best Parameters']]

            # Round numerical columns for neatness
            results_df = results_df.round(2)

            # Reset the index for a clean DataFrame
            results_df.reset_index(drop=True, inplace=True)

        return results_df, classifier
    except Exception as e:
        raise CustomException(e, sys)
        
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)