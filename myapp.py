import pandas as pd
import pickle as pk 
import streamlit as st

# Load model
model = pk.load(open(r'C:\Users\dharam\Desktop\House_price_prediction\House_model_prediction.pkl', 'rb'))

st.header('🏡 Bangalore House Price Predictor')

# Load dataset
data = pd.read_csv(r'C:\Users\dharam\Desktop\House_price_prediction\Cleaned_data.csv')

# Inputs
loc = st.selectbox('Choose the Location', data['location'].unique())
sqft = st.number_input('Enter Total Square Feet')
beds = st.number_input('Enter No. of Bedrooms')
bath = st.number_input('Enter No. of Bathrooms')
balc = st.number_input('Enter No. of Balconies')

# Prepare input DataFrame
input_data = pd.DataFrame([[loc, sqft, bath, balc, beds]], 
                          columns=['location','total_sqft','bath','balcony','bedrooms'])

# Prediction
if st.button("Predict Price"):
    output = model.predict(input_data)
    out_str = '💰 Estimated Price of the House: ₹ ' + str(round(output[0] * 100000, 2))
    st.success(out_str)
