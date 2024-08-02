import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load models
with open('KNN_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)
with open('Naive_model.pkl', 'rb') as file:
    naive_model = pickle.load(file)

# Initialize label encoders
label_encoders = {
    'Sex': LabelEncoder(),
    'ChestPainType': LabelEncoder(),
    'RestingECG': LabelEncoder(),
    'ExerciseAngina': LabelEncoder(),
    'ST_Slope': LabelEncoder()
}

# Example data to fit label encoders
# Replace with actual fitting on training data
label_encoders['Sex'].fit(['Male', 'Female'])
label_encoders['ChestPainType'].fit(['ATA', 'NAP', 'ASY', 'TA'])
label_encoders['RestingECG'].fit(['Normal', 'ST', 'LVH'])
label_encoders['ExerciseAngina'].fit(['Y', 'N'])
label_encoders['ST_Slope'].fit(['Up', 'Flat', 'Down'])

# Define the feature list in the same order as training
features = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

# Function to preprocess user input
def preprocess_input(input_data):
    # Encode categorical features
    encoded_data = [
        label_encoders['Sex'].transform([input_data[1]])[0],
        label_encoders['ChestPainType'].transform([input_data[2]])[0],
        label_encoders['RestingECG'].transform([input_data[3]])[0],
        label_encoders['ExerciseAngina'].transform([input_data[4]])[0],
        label_encoders['ST_Slope'].transform([input_data[5]])[0]
    ]
    
    # Numeric features
    numeric_data = [input_data[0], input_data[3], input_data[4], input_data[6], input_data[7], input_data[8], input_data[9]]
    combined_data = numeric_data + encoded_data

    # Convert to numpy array
    combined_data = np.array(combined_data).reshape(1, -1)

    # Standardize numeric features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
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
