# Importing the necessary library
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData
from xgboost import plot_importance
from src.utils.common import load_object
import pickle


st.set_page_config(page_title="Term Depo`sit Prediction", page_icon="research/DallE logo.png", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)

if 'user_input_data' not in st.session_state:
    st.session_state.user_input_data = pd.DataFrame(columns = ["job", "marital", "education", "housing", "loan", "contact", "month",  "day_of_week", "poutcome", "age", "duration", "campaign", "pdays", "previous", "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"])

@st.cache_data
def load_data():
    
    best_model_path=os.path.join('artifacts','best_model.pkl')
    recall_df_path=os.path.join('artifacts','recall_df.pkl')
    results_df_path=os.path.join('artifacts','results_df.pkl')
    preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
    decision_tree_path=os.path.join('artifacts','Decision_Tree.pkl')
    naive_bayes_path = os.path.join('artifacts', 'Naive_Bayes.pkl')
    logistic_regression_path = os.path.join('artifacts', 'Logistic_Regression.pkl')
    adaboost_path = os.path.join('artifacts', 'AdaBoost_Classifier.pkl')
    random_forest_path = os.path.join('artifacts', 'Random_Forest.pkl')
    svc_path = os.path.join('artifacts', 'Support_Vector_Classifier')






    best_model =load_object(file_path=best_model_path)
    recall_df = load_object(file_path=recall_df_path)
    results_df= load_object(file_path=results_df_path)
    preprocessor=load_object(file_path=preprocessor_path)

    naive_bayes = load_object(file_path=naive_bayes_path)
    logistic_regression = load_object(file_path = logistic_regression_path)
    adaboost = load_object(file_path=adaboost_path)
    random_forest = load_object(file_path=random_forest_path)
    svc = load_object(file_path=svc_path)

    decision_tree = load_object(file_path=decision_tree_path)

    return best_model, recall_df, results_df, preprocessor, decision_tree, naive_bayes, logistic_regression, adaboost, random_forest, svc

best_model, recall_df, results_df, preprocessor, decision_tree, naive_bayes, logistic_regression, adaboost, random_forest, svc = load_data()

with st.sidebar:
    model_name = st.selectbox("Select the model 👇", ["XGBoost Classifier", "Decision Tree", "Naive Bayes", "Logistic Regression", "AdaBoost Classifier", "Random Forest", "Support Vector Classifier"])
    if model_name == 'XGBoost Classifier':
        model = best_model
        st.write('You selected the best model.')
    elif model_name == "Decision Tree":
        model = decision_tree
    elif model_name == "Naive Bayes":
        model = naive_bayes
        st.write("You didn\'t select the best model.")
    elif model_name == "Logistic Regression":
        model = logistic_regression
        st.write("You didn\'t select the best model.")
    elif model_name == "AdaBoost Classifier":
        model = adaboost
        st.write("You didn\'t select the best model.")
    elif model_name == "Random Forest":
        model = random_forest
    elif model_name == "Support Vector Classifier":
        model = svc
    

#st.image("research/DallE logo.png", width=100)
# Main content
st.title("Term Deposit Subscription Prediction")
st.header("Welcome to our banking app! Predict if a client will subscribe to a term deposit.")
st.caption(f"App developed by [Umrav Singh Shekhawat](https://www.linkedin.com/in/umrav-singh-shekhawat/)")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Download Predictions", "Feature Importance", "Models Scores"])

with tab1:
    

    col1, col2, col3 = st.columns([2, 3, 2], gap = "medium")
    with col1:
        st.subheader("Bank Client Details", divider = "rainbow")

        #age = st.slider("What's the age", 27, 98, 38)
        age = st.number_input("What's the age", 27, 98, 38,help="Enter customer's age")
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

    
    
    if st.button("Predict"):   
        data_scaled=preprocessor.transform(custom_df)
        preds = model.predict(data_scaled)
        custom_df['Prediction'] = np.where(preds == 1, 'yes', 'no')
        st.write("### Prediction Result:")
        if preds == 0:
            st.write("❌ Whoops! Customer will most likely not subscribe the term deposit")
        elif preds == 1:
            st.write("✅ Yay! The customer will most likely subscribe the term deposit")
        st.session_state.user_input_data =  pd.concat([st.session_state.user_input_data, custom_df], ignore_index = True)


with tab2:
    st.write("you can view and download upto your last 10 predictions")
    st.write(st.session_state.user_input_data.tail(10))
    st.download_button(label="Download Predictions as CSV", data=st.session_state.user_input_data.to_csv(), file_name='predictions.csv', mime='text/csv')

with tab3:
    if model_name == "XGBoost Classifier":
        fig, ax = plt.subplots()
        plot_importance(model, ax= ax)
        st.pyplot(fig)
    elif model_name in ["Logistic Regression", "Support Vector Classifier", "Naive Bayes"]:
        st.write("Feature importance curve couldn't be generated using XGBoost, our best model. Don't worry, there are plenty of other exciting options to explore! 😊 ")
        
    else:
        feature_importance = model.feature_importances_

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

with tab4:
    df = results_df[["Model", "Test Recall", "Test Precision", "Best Parameters"]].sort_values(by = "Test Recall", ascending = False)
    st.title('Model scores')
    st.table(df)    
    
    

st.markdown("---")
st.write("Developed by Umrav Singh Shekhawat")