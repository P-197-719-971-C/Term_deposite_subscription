# Importing the necessary library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.markdown(
    """
<style>
.css-nzvw1x {
    background-color: #061E42 !important;
    background-image: none !important;
}
.css-1aw8i8e {
    background-image: none !important;
    color: #FFFFFF !important
}
.css-ecnl2d {
    background-color: #496C9F !important;
    color: #496C9F !important
}
.css-15zws4i {
    background-color: #496C9F !important;
    color: #FFFFFF !important
}
</style>
""",
    unsafe_allow_html=True
)

# Wrap your entire app in a div with the custom class
st.markdown('<div class="custom-container">', unsafe_allow_html=True)

# Rest of your Streamlit app code goes here...


with st.sidebar:
    st.radio("Model",["Xgboost", "Logistic Regression"])
   


tab1, tab2, tab3 = st.tabs(["Predict", "Feature Importance", "Evaluation Metric"])

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

    if st.button("Predict if the client will subscribe a term deposit?"):   
        st.write("### Prediction Result:")
        st.write(prediction_result)

with tab2:
    chart_data = pd.DataFrame(
   {
       "col1": list(range(20)) * 3,
       "col2": np.random.randn(60),
       "col3": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
   }
    )

    st.bar_chart(chart_data, x="col1", y="col2", color="col3")
with tab3:
    data = {
    'Column1': [1, 2, 3, 4, 5],
    'Column2': [5, 4, 3, 2, 1]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Streamlit app
    st.title('Line Chart for Two Columns')

    # Select columns for the line chart
    selected_columns = st.multiselect('Select Columns:', df.columns)

    # Check if at least two columns are selected
    if len(selected_columns) >= 2:
        # Create a line chart using selected columns
        plt.figure(figsize=(8, 6))
        for column in selected_columns:
            plt.plot(df.index, df[column], marker='o', label=column)
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        st.pyplot()

    else:
        st.warning('Please select at least two columns for the line chart.')

# Close the wrapping div tag
st.markdown('</div>', unsafe_allow_html=True)