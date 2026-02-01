# Path: src/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# 1. Setup Project Path to import modules
sys.path.append(os.path.abspath(os.path.join('..')))

# Import our custom engines
from src.feature_engineering import preprocess_data, split_data
from src.model_trainer import train_churn_model

# --- Page Configuration ---
st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

st.title("🔮 Telco Customer Churn Predictor")
st.markdown("Use Artificial Intelligence to predict if a customer is likely to leave.")

# --- 2. Load & Train Model (Cached) ---
@st.cache_data
def load_and_train():
    # Load raw data
    df = pd.read_csv('data/raw/telecom_churn_v1.csv')
    
    # Preprocess and Split
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train Model
    model = train_churn_model(X_train, y_train)
    return model, df

# Load model and data once
model, df_raw = load_and_train()

# --- 3. Sidebar: User Inputs ---
st.sidebar.header("👤 Customer Profile")

def user_input_features():
    # Input Sliders and Selectboxes
    tenure = st.sidebar.slider('Tenure (Months)', 1, 72, 12)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 70.0)
    contract = st.sidebar.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    internet = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    
    # Manually map inputs to One-Hot Encoded features
    # Note: 'Month-to-month' and 'DSL' are the baseline (dropped columns)
    data = {
        'Tenure_Months': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet == 'No' else 0
    }
    
    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Ensure columns match the model's training data exactly
    # Fill any missing columns with 0
    for col in model.feature_names_in_:
        if col not in features.columns:
            features[col] = 0
            
    # Reorder columns to match model
    features = features[model.feature_names_in_]
    
    return features

input_df = user_input_features()

# --- 4. Main Panel: Prediction & Insights ---

# Row 1: Prediction Result
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Data Review")
    st.write(input_df)

with col2:
    st.subheader("Prediction Result")
    
    # Get Probability
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    churn_prob = probability[0][1] # Probability of 'Yes' (1)
    
    # Display Logic
    if churn_prob > 0.5:
        st.error(f"🚨 High Churn Risk! Probability: {churn_prob*100:.1f}%")
        st.write("**Recommendation:** Offer a 20% discount or a 1-year contract upgrade immediately.")
    else:
        st.success(f"✅ Safe Customer. Probability: {churn_prob*100:.1f}%")
        st.write("**Recommendation:** Customer is loyal. No immediate action needed.")

st.markdown("---")

# Row 2: Visualizations
st.subheader("📊 Strategic Insights")

c1, c2 = st.columns(2)

with c1:
    # Histogram: Monthly Charges
    fig_charges = px.histogram(df_raw, x="MonthlyCharges", color="Churn", 
                               title="Churn Risk by Monthly Charges", 
                               barmode="overlay", opacity=0.7,
                               color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"})
    st.plotly_chart(fig_charges, use_container_width=True)

with c2:
    # Histogram: Contract Type
    fig_contract = px.histogram(df_raw, x="Contract", color="Churn", 
                                title="Churn Risk by Contract Type", 
                                barmode="group",
                                color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"})
    st.plotly_chart(fig_contract, use_container_width=True)