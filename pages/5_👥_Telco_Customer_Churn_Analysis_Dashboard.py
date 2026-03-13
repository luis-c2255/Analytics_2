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


# Title
st.markdown(
    Components.page_header(
        "👥 Telco Customer Churn Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("   ") 
st.subheader("🏠 :orange[Overview]", divider="orange")
st.markdown("   ") 
# Key Metrics Row  
col1, col2, col3, col4 = st.columns(4)  

total_customer = len(df)
churned_customers = df['Churn_Binary'].sum()
churn_rate = (churned_customers / total_customers) * 100
avg_revenue = df['MonthlyCharges'].mean()

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Customers",
            value=f"{total_customers:,}",
            delta="",
            card_type="info",
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Churned Customers",
            value=f"{churned_customers:,}",
            delta=f"-{churn_rate:.1f}%",
            card_type="error"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Avg Monthly Revenue",
            value=f"${avg_revenue:.2f}",
            delta="",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col4:
    revenue_lost = df[df['Churn_Binary']==1]['MonthlyCharges'].sum()
    st.markdown(
        Components.metric_card(
            title="Monthly Revenue Lost",
            value=f"${revenue_lost:,.0f}",
            delta="",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader("📉 :red[Churn Distribution]")
churn_counts = df['Churn'].value_counts()
fig = px.pie(
    values=churn_counts.values,
    names=['Retained', 'Churned'],
    color_discrete_sequence=['#66c2a5', '#fc8d62'],
    hole=0.4
)
fig.update_layout(height=400)
st.plotly_chart(fig, width="stretch")

st.markdown("   ")
st.subheader("📊 :red[Customer Distribution by Contract Type]")
contract_data = df['Contract'].value_counts()
fig2 = px.bar(
    x=contract_data.index,
    y=contract_data.values, 
    labels={'x': 'Contract Type', 'y': 'Number of Customers'},
    color=contract_data.values, 
    color_continuous_scale='Blues'
)
fig2.update_layout(height=400, showlegend=False) 
st.plotly_chart(fig2, width="stretch")  

# Tenure vs Monthly Charges Scatter  
st.markdown("   ")
st.subheader("📈 :red[Tenure vs Monthly Charges by Churn Status]")
fig3 = px.scatter(
    df,
    x='tenure',
    y='MonthlyCharges',
    color='Churn',
    color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'},
    opacity=0.6, 
    labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)'},
    hover_data=['Contract', 'InternetService']
)
fig3.update_layout(height=500)
st.plotly_chart(fig3, width="stretch") 

st.subheader("📈 :blue[Churn Analysis]", divider="blue")

st.subheader("💰 :yellow[Revenue Impact]", divider="yellow")

st.subheader("🤖 :violet[Churn Predictor]", divider="violet")

st.subheader("📊 :rainbow[Customer Segmentation]", divider="rainbow")
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
