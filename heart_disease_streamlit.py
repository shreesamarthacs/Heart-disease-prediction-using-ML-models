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

# Load the fitted encoder and scaler
with open('one_hot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=[
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    ])

    categorical_columns = input_df.select_dtypes(include=['object']).columns.tolist()

    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.transform(input_df[categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_skencoded = pd.concat([input_df, one_hot_df], axis=1)

# Drop the original categorical columns
    df_skencoded = df_skencoded.drop(categorical_columns, axis=1)

    st.dataframe(df_skencoded)

    # Standardize numeric features
    scaled_data = scaler.transform(df_skencoded)
    
    return scaled_data

st.title('Heart Disease Prediction')

# User input
age = st.number_input('Age', min_value=0)
sex = st.selectbox('Sex', options=['M', 'F'])
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
