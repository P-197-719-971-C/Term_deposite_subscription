import os
import pandas as pd
import numpy as np
import pickle
import joblib
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData
import time
import requests

from functools import lru_cache

@lru_cache(maxsize=None)
def load_data():
    obj = DataIngestion()
    train_data, valid_data, test_data, raw_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    df_arr, train_arr, valid_arr, test_arr, df_arr_encoded, train_arr_encoded, valid_arr_encoded, test_arr_encoded,_ = data_transformation.initiate_data_transformation(train_data, valid_data, test_data, raw_data)

    modeltrainer = ModelTrainer()
    recall_df, best_model_name, best_model, results_df, classifier = modeltrainer.initiate_model_trainer(df_arr, train_arr, valid_arr, test_arr, df_arr_encoded, train_arr_encoded, valid_arr_encoded, test_arr_encoded)
    return recall_df, best_model_name, best_model, results_df, classifier

recall_df, best_model_name, best_model, results_df, classifier = load_data()

logging.info("Saving the recall_df.joblib, best_model_name.joblib, best_model.joblib, results_df.joblib, classifier.joblib")
# Create the 'artifacts' directory if it doesn't exist
os.makedirs('artifacts', exist_ok=True)

# Save each variable to a separate file within the 'artifacts' directory
with open('artifacts/recall_df.joblib', 'wb') as file:
    joblib.dump(recall_df, file)

with open('artifacts/best_model_name.joblib', 'wb') as file:
    joblib.dump(best_model_name, file)

with open('artifacts/best_model.joblib', 'wb') as file:
    joblib.dump(best_model, file)

with open('artifacts/results_df.joblib', 'wb') as file:
    joblib.dump(results_df, file)

with open('artifacts/Decision_Tree.joblib', 'wb') as file:
    joblib.dump(classifier["Decision Tree"], file)

with open('artifacts/Naive_Bayes.joblib', 'wb') as file:
    joblib.dump(classifier["Naive Bayes"], file)

with open('artifacts/Logistic_Regression.joblib', 'wb') as file:
    joblib.dump(classifier["Logistic Regression"], file)

with open('artifacts/AdaBoost_Classifier.joblib', 'wb') as file:
    joblib.dump(classifier["AdaBoost Classifier"], file)

with open('artifacts/Random_Forest.joblib', 'wb') as file:
    joblib.dump(classifier["Random Forest"], file)

with open('artifacts/Support_Vector_Classifier', 'wb') as file:
    joblib.dump(classifier["Support Vector Classifier"], file)

print("Results have been saved in the 'artifacts' directory.")
