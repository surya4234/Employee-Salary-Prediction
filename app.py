import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("enhanced_salary_model.pkl")

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B0082;'>💼 Enhanced Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("#### Fill in employee details to predict income category.")

with st.form("salary_form"):
    age = st.slider("Age", 18, 100, 30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                           'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                                           'Assoc-acdm', 'Assoc-voc', 'Doctorate', '5th-6th', 'Prof-school',
                                           '12th', '1st-4th', '10th', 'Preschool', '7th-8th'])
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                     'Separated', 'Widowed', 'Married-spouse-absent'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                             'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                             'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                             'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                                             'Armed-Forces'])
    relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                                                 'Other-relative', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    gender = st.radio("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", 0, 100000)
    capital_loss = st.number_input("Capital Loss", 0, 100000)
    hours_per_week = st.slider("Hours Per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany',
                                                     'Canada', 'India', 'England', 'Cuba', 'Jamaica',
                                                     'South', 'China', 'Italy', 'Dominican-Republic',
                                                     'Vietnam', 'Guatemala', 'Japan'])
    submitted = st.form_submit_button("🔍 Predict Salary")

if submitted:
    input_dict = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': 100000,
        'education': education,
        'educational-num': 10,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("💰 Likely earns **more than 50K**")
    else:
        st.warning("💼 Likely earns **50K or less**")

    st.info(f"📊 Confidence: {prob:.2%}")
