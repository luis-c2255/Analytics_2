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

total_customers = len(df)
churned_customers = df['Churn_Binary'].sum()
churn_rate = (churned_customers / total_customers) * 100
avg_revenue = df['MonthlyCharges'].mean()

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Customers",
            value=f"{total_customers:,}",
            delta="🔝",
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
            delta="💲",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col4:
    revenue_lost = df[df['Churn_Binary']==1]['MonthlyCharges'].sum()
    st.markdown(
        Components.metric_card(
            title="Monthly Revenue Lost",
            value=f"${revenue_lost:,.0f}",
            delta="❗",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader("📉 :orange[Churn Distribution]")
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
st.subheader("📊 :orange[Customer Distribution by Contract Type]")
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
st.subheader("📈 :orange[Tenure vs Monthly Charges by Churn Status]")
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
# Filter options  
col1, col2, col3 = st.columns(3) 
with col1:
    contract_filter = st.multiselect( 
        "Filter by Contract Type",
        options=df['Contract'].unique(), 
        default=df['Contract'].unique()  
    )
with col2:
    internet_filter = st.multiselect(
        "Filter by Internet Service",
        options=df['InternetService'].unique(),
        default=df['InternetService'].unique()
    )
with col3:
    tenure_range = st.slider(
        "Filter by Tenure (months)",
        min_value=int(df['tenure'].min()),
        max_value=int(df['tenure'].max()), 
        value=(int(df['tenure'].min()), int(df['tenure'].max()))
    ) 

# Apply filters  
filtered_df = df[ 
    (df['Contract'].isin(contract_filter)) & 
    (df['InternetService'].isin(internet_filter)) &  
    (df['tenure'] >= tenure_range[0]) &  
    (df['tenure'] <= tenure_range[1])  
]  
st.markdown(f"**Filtered Dataset: {len(filtered_df):,} customers**")  
st.markdown("   ")  

st.subheader("📊 :blue[Churn Rate by Contract Type]")  
contract_churn = filtered_df.groupby('Contract')['Churn_Binary'].mean() * 100 
fig4 = px.bar(
    x=contract_churn.index,
    y=contract_churn.values,
    labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
    color=contract_churn.values,
    color_continuous_scale='Reds',  
    text=contract_churn.values.round(1)  
)  
fig4.update_traces(texttemplate='%{text}%', textposition='outside')  
fig4.update_layout(height=400, showlegend=False) 
st.plotly_chart(fig4, width="stretch") 

st.subheader("📊 :blue[Churn Rate by Payment Method]")  
payment_churn = filtered_df.groupby('PaymentMethod')['Churn_Binary'].mean() * 100 
fig5 = px.bar(
    x=payment_churn.values,
    y=payment_churn.index,
    orientation='h', 
    labels={'x': 'Churn Rate (%)', 'y': 'Payment Method'},
    color=payment_churn.values, 
    color_continuous_scale='OrRd',
    text=payment_churn.values.round(1)  
)
fig5.update_traces(texttemplate='%{text}%', textposition='outside')
fig5.update_layout(height=400, showlegend=False) 
st.plotly_chart(fig5, width="stretch")

# Tenure Distribution  
st.subheader("📉 :blue[Tenure Distribution by Churn Status]")  
fig6 = go.Figure() 
fig6.add_trace(go.Histogram(  
    x=filtered_df[filtered_df['Churn']=='No']['tenure'],
    name='Retained', 
    marker_color='#3498db',
    opacity=0.7,
    nbinsx=30  
))
fig6.add_trace(go.Histogram(
    x=filtered_df[filtered_df['Churn']=='Yes']['tenure'],
    name='Churned', 
    marker_color='#e74c3c',  
    opacity=0.7,
    nbinsx=30  
)) 
fig6.update_layout( 
    barmode='overlay',
    xaxis_title='Tenure (months)',
    yaxis_title='Count',
    height=450, 
    legend=dict(x=0.7, y=0.95)  
)
st.plotly_chart(fig6, width="stretch")

# Churn by Services  
st.subheader("🔍 :blue[Churn Rate by Service Usage]") 
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'] 
service_churn_data = []  
for service in services:
    churn_rate = filtered_df.groupby(service)['Churn_Binary'].mean() * 100
for status, rate in churn_rate.items():
    service_churn_data.append({'Service': service, 'Status': status, 'Churn_Rate': rate  
}) 
service_df = pd.DataFrame(service_churn_data)

fig7 = px.bar( 
    service_df,
    x='Service',
    y='Churn_Rate',
    color='Status',
    barmode='group',
    labels={'Churn_Rate': 'Churn Rate (%)', 'Service': 'Service Type'},
    color_discrete_map={'Yes': '#27ae60', 'No': '#e74c3c', 'No internet service': '#95a5a6'}  
) 
fig7.update_layout(height=450)
st.plotly_chart(fig7, width="stretch")

# Key Insights Box  
st.markdown("   ")  
st.subheader("💡 :blue[Key Insights]")
col1, col2, col3 = st.columns(3)  
  
with col1:
    avg_tenure_churned = filtered_df[filtered_df['Churn']=='Yes']['tenure'].mean()  
    avg_tenure_retained = filtered_df[filtered_df['Churn']=='No']['tenure'].mean()  
    st.info(f"**Tenure Impact:**\n\n"
    f"Churned: {avg_tenure_churned:.1f} months\n\n" 
    f"Retained: {avg_tenure_retained:.1f} months")
with col2: 
    monthly_churned = filtered_df[filtered_df['Churn']=='Yes']['MonthlyCharges'].mean() 
    monthly_retained = filtered_df[filtered_df['Churn']=='No']['MonthlyCharges'].mean()  
    st.warning(f"**Monthly Charges:**\n\n"
    f"Churned: ${monthly_churned:.2f}\n\n" 
    f"Retained: ${monthly_retained:.2f}")
with col3:
    high_risk = filtered_df[(filtered_df['Contract'] == 'Month-to-month') & (filtered_df['tenure'] < 12)]  
    high_risk_rate = high_risk['Churn_Binary'].mean() * 100
    st.error(f"**High-Risk Segment:**\n\n"
    f"Month-to-month + Tenure < 12\n\n"
    f"Churn Rate: {high_risk_rate:.1f}%")

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
