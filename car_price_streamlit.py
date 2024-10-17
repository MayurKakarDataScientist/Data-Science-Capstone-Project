import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_filename = 'best_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load the saved feature names used during training
with open('model_features.pkl', 'rb') as file:
    model_features = pickle.load(file)

# Universal preprocessing function to ensure feature compatibility
def preprocess_input(data, model_features):
    # Create a DataFrame from user input
    df = pd.DataFrame([data])
    st.write("Initial DataFrame:", df)  
    
    # Apply one-hot encoding for categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Add missing columns (that were present during training but are not in the input)
    for column in model_features:
        if column not in df.columns:
            df[column] = 0 

    # Ensure the order of columns matches the order expected by the model
    df = df[model_features]

    # Scale numerical features (assuming 'km_driven' is a numerical feature)
    if 'km_driven' in df.columns:
        scaler = StandardScaler()
        df['km_driven'] = scaler.fit_transform(df[['km_driven']])

    return df

# Streamlit app UI
st.title("Car Price Prediction App")

# Input fields for user data
year = st.number_input('Year', min_value=2000, max_value=2024, value=2015)
km_driven = st.number_input('Kilometers Driven', min_value=0, value=10000)
fuel = st.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG'))
name = st.text_input('Car Name', 'Enter the car name')
owner = st.selectbox('Ownership Type', ('First Owner', 'Second Owner', 'Third Owner'))

# Create a dictionary of user input
user_input = {
    'year': year,
    'km_driven': km_driven,
    'fuel': fuel,
    'name': name,
    'owner': owner
}

# Predict button
if st.button('Predict Price'):
    # Preprocess the input and make sure it matches the model's expected features
    preprocessed_data = preprocess_input(user_input, model_features)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)
    
    # Display the prediction
    st.success(f'The predicted selling price of the car is: ₹{prediction[0]:,.2f}')
