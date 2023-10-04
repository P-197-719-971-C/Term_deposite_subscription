import sys
import pandas as pd
from src/BankMarketing.exception.exception import CustomException
from src/BankMarketing.logging.logger import logging
from src/BankMarketing.utils.common import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        job: str,
        marital: str,
        education: str,
        default: str,
        loan: str,
        contact: str,
        month: str,
        day_of_week: str,
        poutcome: str,
        
        age: int,
        duration: int,
        campaign: int,
        pdays: int,
        previous: int,
        emp.var.rate: float,
        cons_price_idx: float,
        cons_conf_idx: float,
        euribor3m: float,
        nr_employed: float) -> None:

        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.loan = loan
        self.contact = contact
        self.month = month
        self.day_of_week = day_of_week
        self.poutcome = poutcome    
        
        self.age = age
        self.duration = duration
        self.campaign = campaign
        self.pdays = pdays
        self.previous = previous
        self.emp_var_rate = emp.var.rate
        self.cons_price_idx = cons.price.idx
        self.cons_conf_idx = cons.conf.idx   
        self.euribor3m = euribor3m  
        self.nr_employed = nr.employed



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "job": [self.job],
                "marital": [self.marital],
                "education": [self.education],
                "default": [self.default],
                "loan": [self.loan],
                "contact": [self.contact],
                "month": [self.month],
                "day_of_week": [self.day_of_week],
                "poutcome": [self.poutcome],

                "age": [self.age],
                "duration": [self.duration],
                "campaign": [self.campaign],
                "pdays": [self.pdays],
                "previous": [self.previous],
                "emp.var.rate": [self.emp.var.rate],
                "cons.price.idx": [self.cons.price.idx],
                "cons.conf.idx": [self.cons.conf.idx],
                "euribor3m": [self.euribor3m],
                "nr.employed": [self.nr.employed]


            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)