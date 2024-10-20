
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_filename = 'best_model_Random Forest.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Function to preprocess user input data
def preprocess_input(data):
    # Create a DataFrame from user input
    df = pd.DataFrame([data])
    st.write("Initial DataFrame:", df)

    # Apply one-hot encoding for categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Ensure all necessary columns are included
    expected_columns = [
        'km_driven', 'fuel_Diesel', 'fuel_Petrol', 'fuel_CNG',
        'transmission_Automatic', 'transmission_Manual', 'owner'
    ]

    # Create missing columns with default values (0)
    for column in expected_columns:
        if column not in df.columns:
            df[column] = 0

    # Reorder the DataFrame to match the expected column order
    df = df[expected_columns]

    # Scale numerical features
    scaler = StandardScaler()
    df['km_driven'] = scaler.fit_transform(df[['km_driven']])

    return df

# Streamlit app UI
st.title("Car Price Prediction App")

# Input fields for user data
name = st.text_input('Car Name', 'Enter the car name')
km_driven = st.number_input('Kilometers Driven', min_value=0, value=10000)
fuel = st.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG'))
transmission = st.selectbox('Transmission', ('Manual', 'Automatic'))
owner = st.number_input('Number of Owners', min_value=0, value=1)

# Create a dictionary of user input
user_input = {
    'Name': name,
    'km_driven': km_driven,
    'fuel': fuel,
    'transmission': transmission,
    'owner': owner
}

# Predict button
if st.button('Predict Price'):
    preprocessed_data = preprocess_input(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_data)

    # Display the prediction
    st.success(f'The predicted selling price of the car is: ₹{prediction[0]:,.2f}')
