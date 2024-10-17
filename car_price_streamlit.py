import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_filename = 'best_model_Random Forest.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# List of expected columns based on the model's training features
expected_columns = [
    'year', 'km_driven',
    'fuel_Diesel', 'fuel_Petrol', 'fuel_CNG', 'fuel_Electric', 'fuel_LPG',
    'name_Ambassador Classic 2000 Dsz', 'name_Ambassador Grand 1800 ISZ MPFI PW CL',
    'name_Audi A4 1.8 TFSI', 
    'owner_Second Owner', 'owner_Third Owner', 'owner_Test Drive Car'
]

# Function to preprocess user input data
def preprocess_input(data):
    # Create a DataFrame from user input
    df = pd.DataFrame([data])
    st.write("Initial DataFrame:", df)  
    
    # Apply one-hot encoding for categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Ensure all necessary columns are included
    for column in expected_columns:
        if column not in df.columns:
            df[column] = 0 

    # Reorder the DataFrame to match the expected column order
    df = df[expected_columns]

    # Scale the 'km_driven' feature
    scaler = StandardScaler()
    df['km_driven'] = scaler.fit_transform(df[['km_driven']])

    return df

# Streamlit app UI
st.title("Car Price Prediction App")

# Input fields for user data
year = st.number_input('Year', min_value=2000, max_value=2024, value=2015)
km_driven = st.number_input('Kilometers Driven', min_value=0, value=10000)
fuel = st.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG'))  
name = st.selectbox('Car Name', (
    'Ambassador Classic 2000 Dsz', 'Ambassador Grand 1800 ISZ MPFI PW CL', 
    'Audi A4 1.8 TFSI',  
))  
owner = st.selectbox('Ownership Type', ('First Owner', 'Second Owner', 'Third Owner', 'Test Drive Car'))

# Create a dictionary of user input
user_input = {
    'year': year,
    'km_driven': km_driven,
    'fuel': f'fuel_{fuel}',  
    'name': f'name_{name}',  
    'owner': f'owner_{owner}' 
}

# Predict button
if st.button('Predict Price'):
    preprocessed_data = preprocess_input(user_input)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)
    
    # Display the prediction
    st.success(f'The predicted selling price of the car is: ₹{prediction[0]:,.2f}')
