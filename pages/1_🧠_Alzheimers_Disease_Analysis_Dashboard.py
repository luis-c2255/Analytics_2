import streamlit as st  
import pandas as pd  
import numpy as np  
import plotly.express as px  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
import pickle  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
import warnings  
warnings.filterwarnings('ignore')  

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Alzheimers Disease Analysis Dashboard", "🧠")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('alzheimers_disease_data.csv')
    df['Diagnosis_Label'] = df['Diagnosis'].map({0: 'No Alzheimers', 1: 'Alzheimers'})  
    df['Gender_Label'] = df['Gender'].map({0: 'Male', 1: 'Female'})  
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 65, 75, 85, 100],  
    labels=['<65', '65-75', '75-85', '85+'])
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],  
    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])  
    df['Cognitive_Status'] = pd.cut(df['MMSE'], bins=[0, 10, 18, 24, 30],  
    labels=['Severe', 'Moderate', 'Mild', 'Normal'])  
    symptom_cols = ['MemoryComplaints', 'BehavioralProblems', 'Confusion',  
    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',  
    'Forgetfulness']  
    df['Total_Symptoms'] = df[symptom_cols].sum(axis=1)  
  
    return df  
# Train model  
@st.cache_resource  
def train_model(df):  
    feature_columns = [col for col in df.columns if col not in  
    ['PatientID', 'Diagnosis', 'DoctorInCharge', 'Diagnosis_Label',  
    'Gender_Label', 'Age_Group', 'BMI_Category', 'Cognitive_Status']]  
  
    X = df[feature_columns].select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).median())  
    y = df['Diagnosis']  
  
    model = RandomForestClassifier(n_estimators=100, random_state=42)  
    model.fit(X, y)  
  
    scaler = StandardScaler()  
    scaler.fit(X)  
  
    return model, scaler, X.columns.tolist()    
# Title
st.markdown(
    Components.page_header(
        "🧠 Alzheimers Disease Analysis Dashboard"
    ), unsafe_allow_html=True
)
# Load data
df = load_data()
model, scaler, feature_names = train_model(df)


st.markdown(
    Components.section_header("Dataset Overview", "📋"), unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Total Patients",
            value=f"{len(df):,}",
            delta="🙍",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    ad_count = df['Diagnosis'].sum()
    st.markdown(
        Components.metric_card(
            title="Alzheimer's Cases",
            value=f"{ad_count:,}",
            delta=f"{ad_count/len(df)*100:.1f}%",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    avg_age = df['Age'].mean()
    st.markdown(
        Components.metric_card(
            title="Average Age",
            value=f"{avg_age:.1f} years",
            delta="🎂",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    avg_mmse = df['MMSE'].mean()
    st.markdown(
        Components.metric_card(
            title="Average MMSE",
            value=f"{avg_mmse:.1f}",
            delta="📑",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("---")
st.subheader("📊 Diagnosis Distribution")
with st.container():
    fig = go.Figure(
        data=[go.Pie(
            labels=df['Diagnosis_Label'].value_counts().index,
            values=df['Diagnosis_Label'].value_counts().values,
            hole=0.4,
            marker_colors=['#2ecc71', '#e74c3c']
        )])
    fig.update_layout(height=400)
    fig = apply_chart_theme(fig)
    st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.subheader("👥 Age Distribution by Diagnosis")
with st.container():
    fig2 = go.Figure()
    for diagnosis in df['Diagnosis_Label'].unique():
        subset = df[df['Diagnosis_Label'] == diagnosis]
    fig2.add_trace(
        go.Histogram(
            x=subset['Age'],
            name=diagnosis,
            opacity=0.7,
            nbinsx=20
        ))
    fig2.update_layout(barmode='overlay', height=400)
    fig2 = apply_chart_theme(fig2)
    st.plotly_chart(fig2, width="stretch")

st.markdown("---")

# Dataset preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(100), height=300)

# Download option
csv = df.to_csv(index=False)
st.download_button(
label="📥 Download Full Dataset",
data=csv,
file_name="alzheimers_data.csv",
mime="text/csv"
)

st.markdown(
    Components.page_header(
        "🔬 Exploratory Data Analysis"
    ), unsafe_allow_html=True
)
analysis_type = st.selectbox(
    "Select Analysis Type:",
["Demographics", "Clinical Markers", "Lifestyle Factors",
"Symptom Analysis", "Correlation Analysis"]
)

if analysis_type == "Demographics":
    st.subheader("👥 Demographic Analysis")

with st.container():
    gender_diagnosis = df.groupby(['Gender_Label', 'Diagnosis_Label']).size().unstack()
    fig3 = go.Figure()
    for diagnosis in gender_diagnosis.columns:
        fig3.add_trace(go.Bar(
                name=diagnosis,
                x=gender_diagnosis.index,
                y=gender_diagnosis[diagnosis]
            ))
        fig3.update_layout(
            title='Gender Distribution by Diagnosis',
            barmode='group',
            height=400
        )
        fig3 = apply_chart_theme(fig3)
        st.plotly_chart(fig3, width="stretch")

st.markdown("---")    
with st.container():
    age_diagnosis = df.groupby(['Age_Group', 'Diagnosis_Label']).size().unstack()
    fig4 = go.Figure()
    for diagnosis in age_diagnosis.columns:
        fig4.add_trace(go.Bar(
            name=diagnosis,
            x=age_diagnosis.index,
            y=age_diagnosis[diagnosis]
        ))
    fig4.update_layout(
        title='Age Group Distribution by Diagnosis',
        barmode='group',
        height=400
    )
    fig4 = apply_chart_theme(fig4)
    st.plotly_chart(fig4, width="stretch")
# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🧠 Alzheimers Disease Analysis Dashboard</strong></p>
    <p>Patients data analysis</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
