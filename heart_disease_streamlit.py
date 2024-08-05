import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load models
with open('KNN_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)
with open('Naive_model.pkl', 'rb') as file:
    naive_model = pickle.load(file)

# Load and fit OneHotEncoder
dataset_raw = pd.read_csv('heart.xls')
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
one_hot_encoder.fit(dataset_raw[categorical_features])

# Define the feature list
features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=features)
    
    # One-hot encode categorical features
    categorical_data = input_df[categorical_features]
    numeric_data = input_df.drop(columns=categorical_features)
    
    one_hot_encoded = one_hot_encoder.transform(categorical_data)
    
    # Concatenate numeric data and one-hot encoded data
    processed_data = np.concatenate([numeric_data, one_hot_encoded], axis=1)
    
    # Standardize numeric features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(processed_data)
    return scaled_data

st.title('Heart Disease Prediction')

# User input
age = st.number_input('Age', min_value=0)
sex = st.selectbox('Sex', options=['Male', 'Female'])
chest_pain_type = st.selectbox('Chest Pain Type', options=['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input('Resting Blood Pressure', min_value=0)
cholesterol = st.number_input('Cholesterol', min_value=0)
fasting_bs = st.number_input('Fasting Blood Sugar', min_value=0)
resting_ecg = st.selectbox('Resting ECG', options=['Normal', 'ST', 'LVH'])
max_hr = st.number_input('Max Heart Rate', min_value=0)
exercise_angina = st.selectbox('Exercise Angina', options=['Y', 'N'])
oldpeak = st.number_input('Oldpeak', min_value=0.0)
st_slope = st.selectbox('ST Slope', options=['Up', 'Flat', 'Down'])

# Prepare input data for prediction
input_data = [age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]
processed_input = preprocess_input(input_data)

# Prediction
if st.button('Predict'):
    knn_prediction = knn_model.predict(processed_input)
    logistic_prediction = logistic_model.predict(processed_input)
    naive_prediction = naive_model.predict(processed_input)

    st.write(f'KNN Model Prediction: {"Heart Disease" if knn_prediction[0] == 1 else "No Heart Disease"}')
    st.write(f'Logistic Regression Prediction: {"Heart Disease" if logistic_prediction[0] == 1 else "No Heart Disease"}')
    st.write(f'Naive Bayes Prediction: {"Heart Disease" if naive_prediction[0] == 1 else "No Heart Disease"}')
