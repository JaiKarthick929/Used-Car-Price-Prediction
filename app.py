import streamlit as st
import pandas as pd
import pickle as pk

# Page Config
st.set_page_config(page_title="CarDekho Price Prediction", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\J\Desktop\GUVI\Project\CarDheko\Data CLeaning\Cleaned_Car_Dheko.csv")
    df1 = pd.read_csv(r"C:\Users\J\Desktop\GUVI\Project\CarDheko\Data CLeaning\Preprocessed_Car_Dheko.csv")
    return df, df1

df, df1 = load_data()

# Helper Function
def inverse_transform(x):
    return x if x == 0 else 1 / x

# Encodings for Categorical Variables
for col in df.columns:
    if df[col].dtype == 'object':
        globals()[col] = dict(zip(df[col].sort_values().unique(), df1[col].sort_values().unique()))

# Title
st.title(" CarDekho Resale Price Prediction")

# Form for User Input
st.subheader("Enter the Car Details for Price Prediction")
with st.form(key='prediction_form'):
    car_model = st.selectbox("**Select Car Model**", df['Car_Model'].unique())
    model_year = st.selectbox("**Select Produced Year**", df['Car_Produced_Year'].unique())
    transmission = st.radio("**Select Transmission Type**", df['Transmission_Type'].unique(), horizontal=True)
    location = st.selectbox("**Select Location**", df['Location'].unique())
    km_driven = st.number_input("**Enter Kilometers Driven**", min_value=0, max_value=int(df['Kilometers_Driven'].max()))
    engine_cc = st.number_input("**Enter Engine CC**", min_value=0, max_value=int(df['Engine_CC'].max()))
    mileage = st.number_input("**Enter Mileage (kmpl)**", min_value=0.0, max_value=float(df['Mileage(kmpl)'].max()))

    # Load Model
    model_path = r"C:\Users\J\Desktop\GUVI\Project\CarDheko\Data CLeaning\GradientBoost_model.pkl"
    best_model = pk.load(open(model_path, 'rb'))

    # Predict Button
    button = st.form_submit_button("**Predict**")
    if button:
        input_data = [[inverse_transform(km_driven), Transmission_Type[transmission], 
                       Car_Model[car_model], model_year, engine_cc, mileage, Location[location]]]
        result = best_model.predict(input_data)
        st.success(f"### **Predicted Car Price: ₹ {result[0]:,.2f}**")
