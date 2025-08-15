import streamlit as st
import pandas as pd
import joblib
import time

# Load model
model = joblib.load("enhanced_salary_model.pkl")

# Custom CSS for enhanced styling and simple fade-in animation
st.markdown("""
    <style>
    .main {
        background: linear-gradient(120deg, #f8fafc 0%, #e0e7ff 100%);
        animation: fade-in 1.2s;
    }
    @keyframes fade-in {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .stButton>button {
        background-color: #4B0082;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 1.2em;
        margin-top: 1em;
        transition: 0.2s ease-in;
    }
    .stButton>button:hover {
        background-color: #5f23b5;
        color: #ececec;
        box-shadow: 0 4px 15px #c6b9e5;
    }
    .stSlider > div[data-baseweb="slider"] {
        color: #4B0082;
    }
    .stSelectbox > div, .stRadio > div {
        background-color: #f3f0fc;
        border-radius: 8px;
        padding: 0.2em 0.7em;
    }
    .stForm {
        background: #f8f4ff;
        border-radius: 16px;
        box-shadow: 0 6px 16px 0 rgba(75,0,130,0.1);
        padding: 2em 2em 1em 2em;
        margin-top: 1em;
        margin-bottom: 2em;
        animation: fade-in 0.8s;
    }
    .stMarkdown h1 {
        color: #4B0082;
        font-size: 2.5em;
        font-weight: bold;
        letter-spacing: 1px;
        margin-bottom: 0.5em;
    }
    .stAlert {
        border-radius: 10px !important;
        font-size: 1.15em;
        animation: fade-in 0.8s;
    }
    .animated-result {
        animation: bounce-in 1s;
    }
    @keyframes bounce-in {
        0% { transform: scale(0.7); opacity: 0.5; }
        60% { transform: scale(1.1); opacity: 1; }
        80% { transform: scale(0.95); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")
st.markdown("<h1 style='text-align: center;'>💼 Enhanced Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #593196;'>Fill in employee details to predict income category.</h4>", unsafe_allow_html=True)

with st.form("salary_form"):
    st.markdown("##### 👤 Demographics")
    age = st.slider("Age", 18, 100, 30)
    gender = st.radio("Gender", ['Male', 'Female'], horizontal=True)
    race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany',
                                                     'Canada', 'India', 'England', 'Cuba', 'Jamaica',
                                                     'South', 'China', 'Italy', 'Dominican-Republic',
                                                     'Vietnam', 'Guatemala', 'Japan'])

    st.markdown("##### 💼 Work & Education")
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
    hours_per_week = st.slider("Hours Per Week", 1, 100, 40)

    st.markdown("##### 💲 Financial")
    capital_gain = st.number_input("Capital Gain", 0, 100000)
    capital_loss = st.number_input("Capital Loss", 0, 100000)

    submitted = st.form_submit_button("🔍 Predict Salary")

if submitted:
    with st.spinner("⏳ Running prediction and updating transactions..."):
        # Simulate processing and transaction delay for a more dynamic feel
        time.sleep(1.1)
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
        time.sleep(0.7)  # extra pause for effect

    # Animation for showing results
    st.markdown('<div class="animated-result">', unsafe_allow_html=True)
    if prediction == 1:
        st.success("💰 <b>Likely earns <span style='color:#4B0082;'>more than 50K</span></b>", icon="✅")
    else:
        st.warning("💼 <b>Likely earns <span style='color:#b30025;'>50K or less</span></b>", icon="⚠️")
    st.info(f"<b>📊 Confidence:</b> <span style='color:#4B0082'>{prob:.2%}</span>", icon="📊", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Simulated transaction history with subtle animation
    st.markdown(
        """
        <div style="margin-top:2.5em; animation: fade-in 1.2s;">
            <h5 style="color:#4B0082;">🧾 Recent Transactions</h5>
            <ul>
                <li>Form submitted for prediction <span style="color:gray;">{}</span></li>
                <li>ML model evaluated candidate <span style="color:gray;">{}</span></li>
                <li>Prediction result displayed <span style="color:gray;">{}</span></li>
            </ul>
        </div>
        """.format(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            time.strftime("%Y-%m-%d %H:%M:%S"),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
        unsafe_allow_html=True,
    )
