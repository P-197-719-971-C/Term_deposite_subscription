import os
import sys
from src.exception.exception import CustomException
from src.logging.logger import logging
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    valid_data_path: str = os.path.join('artifacts', "valid.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")

        try:
            
            logging.info("Read the dataset as dataframe")

            df = pd.read_csv("research/Data/bank-additional-full.csv", delimiter = ";")          
            
            logging.info("Dropping dupicated rows")

            df.drop_duplicates(inplace = True)

            logging.info("Mapping the target column to 1 and 0")

            df['y'] = df['y'].map({'yes': 1, 'no': 0})
            
            logging.info("relacing unknown with Nan Type")

            df = df.replace('unknown', np.NaN)

            logging.info("dropping default column")

            df = df.drop(['default'], axis = 1)

            # Rename specific columns
            df.rename(columns={'emp.var.rate' : "emp_var_rate", 'cons.price.idx' : "cons_price_idx", 'cons.conf.idx' : "cons_conf_idx", 'nr.employed' : "nr_employed"}, inplace=True)


            logging.info("train valid test split")

            temp_train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['y'])
            train_set, valid_set = train_test_split(temp_train_set, test_size=0.25, random_state=42, stratify=temp_train_set['y'])
            logging.info("Making directory for training data")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            os.makedirs(os.path.dirname(self.ingestion_config.valid_data_path), exist_ok= True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok= True)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)

            logging.info("Saving the split datasets to CSV files")
            # Saving the split datasets to CSV files

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            valid_set.to_csv(self.ingestion_config.valid_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.valid_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
    
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, valid_data, test_data, raw_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    preprocessor, encoder = data_transformation.get_data_transformer_object(raw_data)
    df_arr, train_arr, valid_arr, test_arr, df_arr_encoded, train_arr_encoded, valid_arr_encoded, test_arr_encoded,_=data_transformation.initiate_data_transformation(train_data, valid_data, test_data, raw_data)

    modeltrainer = ModelTrainer()
    recall_df, best_model_name, best_model, results_df, classifier = modeltrainer.initiate_model_trainer(df_arr, train_arr, valid_arr, test_arr, df_arr_encoded, train_arr_encoded, valid_arr_encoded, test_arr_encoded)
    print(recall_df)
    
