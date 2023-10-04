import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

from src/BankMarketing.exception.exception import CustomException
from src/BankMarketing.logging.logger import logging
from dataclasses import dataclass

from src/BankMarketing.utils.common import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This is responsible for data transformation
        '''
        try:
            numerical_columns = [feature for feature in df.columns if df[feature].dtype != 'O' ]
            categorical_columns = [feature for feature in df.columns if df[feature].dtype == 'O' and feature != 'y']
        

            num_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder", OneHotEncoder()),
                        ("scaler", StandardScaler(with_mean= False))]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            encoder = TargetEncoder()

            return preprocessor, encoder


        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path,valid_path, test_path, raw_path):

        try:
            logging.info("Read train, valid and test data completed")

            train_df = pd.read_csv(train_path)
            valid_df = pd.read_csv(valid_path)
            test_df = pd.read_csv(test_path)
            df = pd.read_csv(raw_path)

            logging.info("obtaining preprocessing object")

            preprocessing_obj, encoder_obj = self.get_data_transformer_object()

            target_column_name = "y"

            numerical_columns = [feature for feature in df.columns if df[feature].dtype != 'O']

            features = df.drop(columns= [target_column_name], axis = 1)
            target = df[target_column_name]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_valid_df = valid_df.drop(columns=[target_column_name], axis = 1)
            target_feature_valid_df = valid_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            X_train_encoded = encoder.fit_transform(input_feature_train_df, target_feature_train_df)
            X_valid_encoded = encoder.transform(input_feature_valid_df, target_feature_valid_df)
            X_test_encoded = encoder.transform(input_feature_test_df, target_feature_test_df)
            X_encoded =  encoder.transform(features, target)

            logging.info(f"Applying preprocessing object on training, valid and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_valid_arr=preprocessing_obj.transform(input_feature_valid_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            X = preprocessing_obj.transform(features)
        
            logging.info(f"Converting to training valid and testing array including target variable")

            df_arr_encoded = np.c[X_encoded, np.array(target)]
            train_arr_encoded = np.c[X_train_encoded, np.array(target_feature_train_df)]
            valid_arr_encoded = np.c_[X_valid_encoded, np.array(target_feature_valid_df)]
            test_arr_encoded = np.c_[X_test_encoded,  np.array(target_feature_test_df)]

            df_arr = np.c_[X, np.array(target)]
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            valid_arr = np.c_[input_feature_valid_arr, np.array(target_feature_valid_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                df_arr,
                train_arr,
                valid_arr,
                test_arr,
                df_arr_encoded,
                train_arr_encoded,
                valid_arr_encoded,
                test_arr_encoded,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e, sys)
