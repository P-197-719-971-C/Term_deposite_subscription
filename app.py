# Importing the necessary library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData


with st.sidebar:
    st.radio("Model",["Xgboost", "Logistic Regression"])



tab1, tab2  = st.tabs(["Predict", "Feature Importance"])

with tab1:
    st.title("Term Deposite Subscription")
    st.divider()
    
    col1, col2, col3 = st.columns([2, 3, 2], gap = "medium")
    with col1:
        st.subheader("Bank Client Details", divider = "rainbow")

        #age = st.slider("What's the age", 27, 98, 38)
        age = st.number_input("What's the age", 27, 98, 38)
        job = st.selectbox("Type of job", ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 'retired', 'management', 'unemployed',
                'self-employed', 'entrepreneur', 'student'], index = 2)
        marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'], index = 0   )
        education = st.selectbox('Education', ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course',
                    'university.degree', 'illiterate'], index = 6)
        housing = st.selectbox( 'Has housing loan?', ['no', 'yes'], index = 1)
        loan = st.selectbox('Has personal loan', ['no', 'yes'], index = 0)
        
    

    with col2:
        st.subheader("Marketing Campaign Details", divider = "rainbow")

        contact = st.select_slider('Contact communication type', ['telephone', 'cellular'], 'cellular')
        month = st.select_slider('Last contact month of year', ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'], 'may')
        day_of_week = st.select_slider('Last contact day of the week', ['mon', 'tue', 'wed', 'thu', 'fri'], 'thu')
        poutcome = st.select_slider('Outcome of the previous marketing campaign', ['nonexistent', 'failure', 'success'], 'nonexistent')

        duration = st.slider("Last call duration (seconds)",0, 4918, 180)
        campaign = st.slider("Number of contacts performed during the current campaign and for this client", 1, 56, 2)
        pdays = st.slider("Number of days that passed by after the client was last contacted from a previous campaign", 0, 999, 999)
        previous = st.slider("Number of contacts performed before this campaign and for this client", 0.00, 7.00, 0.17)

    with col3:
        st.subheader("Global Variables", divider = "rainbow")
        emp_var_rate = st.number_input("employment variation rate - quarterly indicator", -3.4, 1.4, 1.1 )
        cons_price_idx = st.number_input("consumer price index - monthly indicator", 92.2, 94.8, 93.7)
        cons_conf_idx = st.number_input("consumer confidence index - monthly indicator", -50.0, -27.0, -41.8)
        euribor3m = st.number_input("euribor 3 month rate - daily indicator", 0.634, 5.045, 4.857)
        nr_employed = st.number_input("number of employees - quarterly indicator", 4963, 5228, 5191)

    custom_inputs = CustomData(job, marital, education, loan, contact, month, day_of_week, poutcome, age, duration, campaign,
                                pdays, previous, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed)
    custom_df = custom_inputs.get_data_as_data_frame()

    predict_pipeline = PredictPipeline()
    preds = predict_pipeline.predict(custom_df)

    if st.button("Predict if the client will subscribe a term deposit?"):   
        st.write("### Prediction Result:")
        st.write(preds)

with tab2:
    chart_data = pd.DataFrame(
    {
    "col1": list(range(20)) * 3,
    "col2": np.random.randn(60),
    "col3": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
    }
    )

    st.bar_chart(chart_data, x="col1", y="col2", color="col3")
 

