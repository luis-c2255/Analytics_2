import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  
import joblib  
import seaborn as sns  
import matplotlib.pyplot as plt 

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Telco Customer Churn Analysis Dashboard", "👥")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')

    # Clean TotalCharges  
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)

    # Encode Churn as binary 
    df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

@st.cache_resource  
def load_model(): 
    try:
        model = joblib.load('churn_prediction_model.pkl')
        features = joblib.load('model_features.pkl') 
        return model, features 
    except:
        return None, None

df = load_data()  
model, model_features = load_model() 

# ========================================  
# SIDEBAR  
# ========================================  
st.sidebar.image("https://img.icons8.com/fluency/96/000000/phone.png", width=80)
st.sidebar.title("📊 Navigation")  

menu = st.sidebar.radio(
    "Select Dashboard View:",
    ["🏠 Overview", "📈 Churn Analysis", "💰 Revenue Impact", 
    "🤖 Churn Predictor", "📊 Customer Segmentation"]  
)  
st.sidebar.markdown("---")  
st.sidebar.markdown("### Dataset Info")  
st.sidebar.metric("Total Customers", f"{len(df):,}")  
st.sidebar.metric("Churn Rate", f"{df['Churn_Binary'].mean()*100:.2f}%")
st.sidebar.metric("Total Revenue", f"${df['TotalCharges'].sum():,.0f}") 

# Title
st.markdown(
    Components.page_header(
        "👥 Telco Customer Churn Analysis Dashboard"
    ), unsafe_allow_html=True
)
# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>👥 Telco Customer Churn Analysis Dashboard</strong></p>
    <p>Analysis of Telco Customer Churn dataset</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
