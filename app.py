# Importing the necessary library
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData
from xgboost import plot_importance
from src.utils.common import load_object
st.set_page_config(page_title="Term Deposit Prediction", page_icon="✅", layout="wide")

@st.cache_data
def load_data():
    best_model_name_path =os.path.join("artifacts","best_model_name.pkl")
    best_model_path=os.path.join('artifacts','best_model.pkl')
    classifier_path=os.path.join('artifacts','classifier.pkl')
    recall_df_path=os.path.join('artifacts','recall_df.pkl')
    results_df_path=os.path.join('artifacts','results_df.pkl')
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

    best_model_name=load_object(file_path=best_model_name_path)
    best_model =load_object(file_path=best_model_path)
    classifier=load_object(file_path=classifier_path)
    recall_df=load_object(file_path=recall_df_path)
    results_df=load_object(file_path=results_df_path)
    preprocessor=load_object(file_path=preprocessor_path)
    return best_model_name, best_model, classifier, recall_df, results_df, preprocessor

best_model_name, best_model, classifier, recall_df, results_df, preprocessor = load_data()

with st.sidebar:
    model_name = st.selectbox("Select the model 👇", ["XGBoost Classifier", "Decision Tree", "Naive Bayes", "Logistic Regression", "AdaBoost Classifier", "Random Forest", "Support Vector Classifier"])
    if model_name == 'XGBoost Classifier':
        st.write('You selected the best model.')
    else:
        st.write("You didn\'t select the best model.")

#st.image("research/DallE logo.png", width=100)
# Main content
st.title("Term Deposit Subscription Prediction")
st.header("Welcome to our banking app! Predict if a client will subscribe to a term deposit.")

tab1, tab2, tab3 = st.tabs(["Predict", "Feature Importance", "Models Scores"])

with tab1:
    
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

    custom_inputs = CustomData(job, marital, education, housing, loan, contact, month, day_of_week, poutcome, age, duration, campaign,
                                pdays, previous, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed)
    custom_df = custom_inputs.get_data_as_data_frame()

    """predict_pipeline = PredictPipeline()
    preds = predict_pipeline.predict(custom_df)"""
    data_scaled=preprocessor.transform(custom_df)
    preds = classifier[model_name].predict(data_scaled)
    
    if st.button("Predict"):   
        st.write("### Prediction Result:")
        if preds == 0:
            st.write("❌ Whoops! Customer will most likely not subscribe the term deposit")
        elif preds == 1:
            st.write("✅ Yay! The customer will most likely subscribe the term deposit")

with tab2:
    if model_name == "XGBoost Classifier":
        fig, ax = plt.subplots()
        plot_importance(classifier[model_name], ax= ax)
        st.pyplot(fig)
    elif model_name in ["Logistic Regression", "Support Vector Classifier", "Naive Bayes"]:
        st.write("Feature importance curve couldn't be generated using XGBoost, our best model. Don't worry, there are plenty of other exciting options to explore! 😊 ")
        
    else:
        feature_importance = classifier[model_name].feature_importances_

        # Create a DataFrame for visualization
        feature_importance_df = pd.DataFrame({'Importance': feature_importance})

        # Streamlit app
        st.title('Feature Importance Plot')

        # Plot feature importance without feature labels
        st.write("Feature Importance Plot:")
        fig, ax = plt.subplots()
        ax.barh(range(len(feature_importance)), feature_importance)
        ax.set_xlabel('Importance')
        ax.set_yticks([])  # Remove y-axis ticks and labels
        st.pyplot(fig)

with tab3:
    df = results_df[["Model", "Test Recall", "Test Precision", "Best Parameters"]].sort_values(by = "Test Recall", ascending = False)
    st.title('Model scores')
    st.dataframe(df)    
    
    

st.markdown("---")
st.write("Developed by Umrav Singh Shekhawat")