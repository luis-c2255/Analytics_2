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
    st.plotly_chart(fig4, width="stretch")

# Education &  Ethnicity
# BMI categories
st.markdown("---")
with st.container():
    bmi_diagnosis = df.groupby(['BMI_Category', 'Diagnosis_Label']).size().unstack() 
    fig_bmi = px.bar(
        bmi_diagnosis,
        barmode='group',
        title='BMI Category by Diagnosis')
    fig_bmi.update_layout(height=400)
    st.plotly_chart(fig_bmi, width="stretch")
st.markdown("---")
# Cognitive status
with st.container():
    cog_diagnosis = df.groupby(['Cognitive_Status', 'Diagnosis_Label']).size().unstack()
    fig_cog = px,bar(
        cog_diagnosis,
        barmode='group',
        title='Cognitive Status by Diagnosis')
    fig_cog.update_layout(height=400)
    st.plotly_chart(fig_cog, width="stretch")

elif: 
    analysis_type == 'Clinical Markers':
    st.subheader("🏥 Clinical Markers Analysis")  

    # Select clinical markers
    clinical_markers = st.multiselect(  
        "Select Clinical Markers:",  
        ['MMSE', 'FunctionalAssessment', 'ADL', 'SystolicBP', 'DiastolicBP',  
        'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'BMI'],  
        default=['MMSE', 'FunctionalAssessment', 'ADL']  
    )  
# Box plots for selected markers  
if clinical_markers:
    cols = st.columns(min(len(clinical_markers),3))


    for idx, marker in enumerate(clinical_markers):
        with cols[idx % 3]:
            fig_mark = go.Figure()
            for diagnosis in df['Diagnosis_Label'].unique():
                subset = df[df['Diagnosis_Label'] == diagnosis]
            fig_mark.add_trace(go.Box(
                y=subset[marker],
                name=diagnosis,
                boxmean='sd'
            ))
            fig_mark.update_layout(
                title=f"{marker}Distribution",
                height=350
            )
            st.plotly_chart(fig_mark, width="stretch")
# Statistical comparison
st.markdown("---")  
st.subheader("📊 Statistical Comparison") 

comparison_data = []
for marker in clinical_markers:
    no_ad = df[df['Diagnosis'] == 0][marker].dropna()
    ad = df[df['Diagnosis'] == 1][marker].dropna()

    comparison_data.append({
        'Marker': marker,
        'No AD (Mean)': f"{no_ad.mean():.2f}",
        'AD (Mean)': f"{ad.mean():.2f}",
        'Difference': f"{ad.mean() - no_add.mean():.2f}",
        'No AD (Std)': f"{no_ad.std():.2f}", 
        'AD (Std)': f"{ad.std():.2f}"   
    })
    st.dataframe(pd.DataFrame(comparison_data), width="stretch")  
    elif: 
        analysis_type == "Lifestyle Factors": 
        st.subheader("🏃 Lifestyle Factors Analysis")
        
        col1, col2 = st.columns(2) 
    with col1:
        fig_act = go.Figure()
        for diagnosis in df['Diagnosis_Label'].unique():
            subset = df[df['Diagnosis_Label'] == diagnosis]
            fig_act.add_trace(go.Violin(
                y=subset['PhysicalActivity'],
                name=diagnosis,
                box_visible=True,
                meanline_visible=True
            ))
            fig_act.update_layout(
                title="Physical Activity by Diagnosis",
                yaxis_title="Hours per Week",
                height=400
            )
            st.plotly_chart(fig_act, width="stretch")
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
