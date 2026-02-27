import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.graph_objects as go 
import plotly.express as px
from plotly.subplots import make_subplots 
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.svm import SVC 
from sklearn.metrics import ( confusion_matrix, roc_curve, roc_auc_score, f1_score, classification_report ) 
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu

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
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist() 
    if "PatientID" in numerical_cols: 
        numerical_cols.remove("PatientID") 
        
    imputer = SimpleImputer(strategy="median") 
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Label mappings 
    df["Gender_Label"] = df["Gender"].map({0: "Male", 1: "Female"}) 
    df["Diagnosis_Label"] = df["Diagnosis"].map({0: "No Alzheimers", 1: "Alzheimers"}) 
    ethnicity_map = {0: 'Caucasian', 1:'African American', 2:'Asian', 3:'Other'} 
    education_map = {0: 'None', 1: 'High School', 2: 'Bachelor', 3: 'Higher'} 
    df["Ethnicity_Label"] = df["Ethnicity"].map(ethnicity_map) 
    df["Education_Label"] = df["EducationLevel"].map(education_map) 
    # Derived features 
    df["BP_Category"] = pd.cut(df["SystolicBP"], bins=[0,120,130,140,200],
    labels=["Normal","Elevated","Stage1","Stage2"]) 
    df["BMI_Category"] = pd.cut(df["BMI"], bins=[0,18.5,25,30,100], 
    labels=["Underweight","Normal","Overweight","Obese"]) 
    df["Age_Group"] = pd.cut(df["Age"], bins=[0,65,75,85,100], 
    labels=["<65","65-75","75-85","85+"]) 
    df["Cognitive_Status"] = pd.cut(df["MMSE"], bins=[0,10,18,24,30], 
    labels=["Severe","Moderate","Mild","Normal"]) 
    df["Lifestyle_Risk"] = ( 
        df["Smoking"]*2 + 
        (df["AlcoholConsumption"] > 14).astype(int)*2 + 
        (df["PhysicalActivity"] < 5).astype(int) + 
        (df["DietQuality"] < 5).astype(int) + 
        (df["SleepQuality"] < 6).astype(int) 
    ) 
    symptom_cols = [ 
        'MemoryComplaints','BehavioralProblems','Confusion','Disorientation', 
        'PersonalityChanges','DifficultyCompletingTasks','Forgetfulness' 
    ] 
    df["Total_Symptoms"] = df[symptom_cols].sum(axis=1)
    return df  
  
# Title
st.markdown(
    Components.page_header(
        "🧠 Alzheimers Disease Analysis Dashboard"
    ), unsafe_allow_html=True
)
# Load data
df = load_data()

st.subheader("📋 :orange[Dataset Overview]", divider="orange")

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
st.subheader("📊 :blue[Diagnosis Distribution]", divider="blue")
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
st.subheader("👥 :green[Age Distribution by Diagnosis]", divider="green")
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
st.subheader("🔍 :orange[Dataset Preview]", divider="orange")
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
def plot_grouped_bar_from_crosstab(crosstab_df, title, height=400):
    fig = go.Figure()
    for col in crosstab_df.columns:
        fig.add_trace(
            go.Bar(
                name=col,
                x=crosstab_df.index,
                y=crosstab_df[col],
            )
        )
    fig.update_layout(
        title=title,
        barmode='group',
        height=height,
    )
    st.plotly_chart(fig, width="stretch")

def plot_violin_by_diagnosis(df, value_col, title, yaxis_title, height=400):
    fig = go.Figure()
    for diagnosis in df['Diagnosis_Label'].unique():
        subset = df[df['Diagnosis_Label'] == diagnosis]
        series = subset[value_col].dropna()
        if series.empty:
            continue
        fig.add_trace(
            go.Violin(
                y=series,
                name=diagnosis,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        height=height,
    )
    st.plotly_chart(fig, width="stretch")

analysis_type = st.selectbox(
    "Select Analysis Type:",
    [
        "Demographics",
        "Clinical Markers",
        "Lifestyle Factors",
        "Symptom Analysis",
        "Correlation Analysis",
    ],
)
if analysis_type == "Demographics":
    st.subheader("👥 :red[Demographic Analysis]", divider="red")

    col1, col2 = st.columns(2)

    with col1:
        # Gender distribution
        gender_diagnosis = (
            df.groupby(['Gender_Label', 'Diagnosis_Label'])
            .size()
            .unstack(fill_value=0)
        )
        plot_grouped_bar_from_crosstab(
            gender_diagnosis,
            title="Gender Distribution by Diagnosis",
            height=400,
        )
    
    with col2:
        # Age group distribution
        age_diagnosis = (
            df.groupby(["Age_Group", "Diagnosis_Label"])
            .size()
            .unstack(fill_value=0)
        )
        plot_grouped_bar_from_crosstab(
            age_diagnosis,
            title="Age Group Distribution by Diagnosis",
            height=400,
        )
    
    col1, col2 = st.columns(2)

    with col1:
        #BMI categories
        bmi_diagnosis = (
            df.groupby(["BMI_Category", "Diagnosis_Label"])
            .size()
            .unstack(fill_value=0)
        )
        fig = px.bar(
            bmi_diagnosis,
            barmode="group",
            title='BMI Category by Diagnosis',
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Cognitive status
        cog_diagnosis = (
            df.groupby(["Cognitive_Status", "Diagnosis_Label"])
            .size()
            .unstack(fill_value=0)
        )
        fig = px.bar(
            cog_diagnosis,
            barmode='group',
            title="Cognitive Status by Diagnosis",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
# ============================================
# Clinical Markers
# ============================================
elif analysis_type == "Clinical Markers":
    st.subheader("🏥 :violet[Clinical Markers Analysis]", divider="violet")

    clinical_markers = st.multiselect(
        "Select Clinical Markers:",
        [
            "MMSE",
            "FunctionalAssessment",
            "ADL",
            "SystolicBP",
            "DiastolicBP",
            "CholesterolTotal",
            "CholesterolLDL",
            "CholesterolHDL",
            "BMI",
        ],
        default=["MMSE", "FunctionalAssessment", "ADL"],
    )

    if clinical_markers:
        cols = st.columns(min(len(clinical_markers), 3))

        for idx, marker in enumerate(clinical_markers):
            with cols[idx % 3]:
                fig = go.Figure()
                for diagnosis in df['Diagnosis_Label'].unique():
                    subset = df[df['Diagnosis_Label'] == diagnosis]
                    series = subset[marker].dropna()
                    if series.empty:
                        continue
                    fig.add_trace(
                        go.Box(
                            y=series, 
                            name=diagnosis, 
                            boxmean="sd",
                        )
                    )
                fig.update_layout( 
                    title=f"{marker} Distribution", 
                    height=350, 
                )
                st.plotly_chart(fig, width="stretch")

        # Statistical comparison 
        st.markdown("---") 
        st.subheader("📊 :yellow[Statistical Comparison]", divider="yellow")

        comparison_data = [] 
        for marker in clinical_markers: 
            no_ad = df[df["Diagnosis"] == 0][marker].dropna() 
            ad = df[df["Diagnosis"] == 1][marker].dropna() 
            
            if no_ad.empty or ad.empty: 
                continue
            comparison_data.append( 
                { 
                    "Marker": marker, 
                    "No AD (Mean)": f"{no_ad.mean():.2f}", 
                    "AD (Mean)": f"{ad.mean():.2f}", 
                    "Difference": f"{ad.mean() - no_ad.mean():.2f}", 
                    "No AD (Std)": f"{no_ad.std():.2f}", 
                    "AD (Std)": f"{ad.std():.2f}", 
                } 
            ) 
        if comparison_data: 
            st.dataframe( 
                pd.DataFrame(comparison_data), 
                width="stretch",
            ) 
        else: 
            st.info("Not enough data to compute comparison statistics.")

# ============================================
# Lifestyle Factors
# ============================================

elif analysis_type == "Lifestyle Factors": 
    st.subheader("🏃 :red[Lifestyle Factors Analysis]", divider="red") 
    
    col1, col2 = st.columns(2) 
    
    with col1: 
        # Physical Activity 
        plot_violin_by_diagnosis( 
            df, 
            value_col="PhysicalActivity", 
            title="Physical Activity by Diagnosis", 
            yaxis_title="Hours per Week", 
            height=400, 
        ) 
    with col2: 
        # Diet Quality 
        plot_violin_by_diagnosis( 
            df, 
            value_col="DietQuality", 
            title="Diet Quality by Diagnosis", 
            yaxis_title="Quality Score", 
            height=400, 
        ) 
    
    col1, col2 = st.columns(2) 
    with col1: 
        # Sleep Quality 
        plot_violin_by_diagnosis( 
            df, 
            value_col="SleepQuality", 
            title="Sleep Quality by Diagnosis", 
            yaxis_title="Quality Score", 
            height=400, 
        ) 
    with col2: 
        # Smoking and Alcohol 
        lifestyle_df = df.groupby("Diagnosis_Label")[["Smoking", "AlcoholConsumption"]].mean() 
        fig = go.Figure() 
        fig.add_trace( 
            go.Bar( 
                name="Smoking Rate", 
                x=lifestyle_df.index, 
                y=lifestyle_df["Smoking"] * 100, 
                text=lifestyle_df["Smoking"].apply(lambda x: f"{x * 100:.1f}%"), 
                textposition="auto", 
            ) 
        ) 
        fig.add_trace( 
            go.Bar( 
                name="Avg Alcohol (drinks/week)", 
                x=lifestyle_df.index, y=lifestyle_df["AlcoholConsumption"], 
                text=lifestyle_df["AlcoholConsumption"].apply(lambda x: f"{x:.1f}"), 
                textposition="auto", 
                yaxis="y2", 
            ) 
        ) 
        fig.update_layout( 
            title="Smoking & Alcohol Consumption", 
            yaxis=dict(title="Smoking Rate (%)"), 
            yaxis2=dict( 
                title="Alcohol (drinks/week)", 
                overlaying="y", 
                side="right", 
            ), 
            height=400, 
            barmode="group", 
        ) 
        st.plotly_chart(fig, width="stretch")

# ============================================
# Symptom Analysis
# ============================================
elif analysis_type == "Symptom Analysis": 
    st.subheader("⚠️ :rainbow[Symptom Analysis]", divider="rainbow") 
    
    symptom_cols = [ 
        "MemoryComplaints", 
        "BehavioralProblems", 
        "Confusion", 
        "Disorientation", 
        "PersonalityChanges", 
        "DifficultyCompletingTasks", 
        "Forgetfulness", 
    ] 
    
    # Symptom prevalence 
    symptom_prev = df.groupby("Diagnosis_Label")[symptom_cols].mean() * 100 
    
    fig = go.Figure() 
    for diagnosis in symptom_prev.index: 
        fig.add_trace( 
            go.Bar( 
                name=diagnosis, 
                x=symptom_cols, 
                y=symptom_prev.loc[diagnosis], 
                text=symptom_prev.loc[diagnosis].round(1), 
                textposition="auto", 
            ) 
        ) 
    fig.update_layout( 
        title="Symptom Prevalence (%) by Diagnosis", 
        xaxis_title="Symptoms", 
        yaxis_title="Prevalence (%)", 
        barmode="group", 
        height=500, 
        xaxis_tickangle=-45, 
    ) 
    st.plotly_chart(fig, width="stretch") 
    
    # Total symptom count distribution 
    st.markdown("---") 
    col1, col2 = st.columns(2) 
    
    with col1: 
        fig = go.Figure() 
        for diagnosis in df["Diagnosis_Label"].unique(): 
            subset = df[df["Diagnosis_Label"] == diagnosis] 
            series = subset["Total_Symptoms"].dropna() 
            if series.empty: 
                continue 
            fig.add_trace( 
                go.Histogram( 
                    x=series, 
                    name=diagnosis, 
                    opacity=0.7, 
                ) 
            ) 
        fig.update_layout( 
            title="Total Symptom Count Distribution", 
            xaxis_title="Number of Symptoms", 
            yaxis_title="Frequency", 
            barmode="overlay", 
            height=400, 
        ) 
        st.plotly_chart(fig, width="stretch") 
    
    with col2: 
        # Symptom co-occurrence heatmap 
        symptom_corr = df[symptom_cols].corr() 
        fig = go.Figure( 
            data=go.Heatmap( 
                z=symptom_corr.values, 
                x=symptom_cols, 
                y=symptom_cols, 
                colorscale="RdBu", 
                zmid=0, 
                text=symptom_corr.round(2).astype(str).values, 
                texttemplate="%{text}", 
                textfont={"size": 10}, 
            ) 
        ) 
        fig.update_layout( 
            title="Symptom Co-occurrence Correlation", 
            height=400, 
        ) 
        st.plotly_chart(fig, width="stretch") 

# ------------------------------------------------------------------- 
# Correlation Analysis 
# ------------------------------------------------------------------- 
elif analysis_type == "Correlation Analysis": 
    st.subheader("🔗 :blue[Correlation Analysis]", divider="blue") 
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist() 
    numerical_features = [f for f in numerical_features if f not in ["PatientID"]] 
    
    selected_features = st.multiselect( 
        "Select Features for Correlation Analysis:", 
        numerical_features, 
        default=[ 
            "Age", 
            "MMSE", 
            "FunctionalAssessment", 
            "PhysicalActivity", 
            "SleepQuality", 
            "Total_Symptoms", 
            "Diagnosis", 
        ], 
    ) 
    
    if len(selected_features) > 1: 
        correlation_matrix = df[selected_features].corr() 
        fig = go.Figure( 
            data=go.Heatmap( 
                z=correlation_matrix.values, 
                x=correlation_matrix.columns, 
                y=correlation_matrix.columns, 
                colorscale="RdBu", 
                zmid=0, 
                text=correlation_matrix.round(2).astype(str).values, 
                texttemplate="%{text}", textfont={"size": 10}, 
                colorbar=dict(title="Correlation"), 
            ) 
        ) 
        fig.update_layout( 
            title="Correlation Heatmap", 
            height=600, 
        ) 
        st.plotly_chart(fig, width="stretch") 
        
        # Top correlations with Diagnosis 
        if "Diagnosis" in selected_features: 
            st.markdown("---") 
            st.subheader(":green[Top correlations with Diagnosis]", divider="green") 
            
            corr_with_diag = ( 
                correlation_matrix["Diagnosis"] 
                .drop("Diagnosis") 
                .sort_values(ascending=False) 
            ) 
            corr_df = corr_with_diag.reset_index() 
            corr_df.columns = ["Feature", "Correlation with Diagnosis"] 
            
            st.dataframe(corr_df, width="stretch") 
        else: 
            st.info("Select at least two features to compute correlations.")
st.markdown("---")
ad = df[df["Diagnosis"]==1] 
no_ad = df[df["Diagnosis"]==0]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.metric_card(
            title="MMSE (No AD)",
            value=f"{no_ad['MMSE'].mean():.1f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="MMSE (AD)",
            value=f"{ad['MMSE'].mean():.1f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="MMSE Gap",
            value=f"{no_ad['MMSE'].mean() - ad['MMSE'].mean():.1f}",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.metric_card(
            title="Functional (No AD)",
            value=f"{no_ad['FunctionalAssessment'].mean():.1f}",
            delta="",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Functional (AD)",
            value=f"{ad['FunctionalAssessment'].mean():.1f}",
            delta="",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="ADL Gap",
            value=f"{no_ad['ADL'].mean() - ad['ADL'].mean():.1f}",
            delta="",
            card_type="warning"
        ), unsafe_allow_html=True
    )
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.metric_card(
            title="Systolic BP (AD)",
            value=f"{ad['SystolicBP'].mean():.1f}",
            delta="",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Diastolic BP (AD)",
            value=f"{ad['DiastolicBP'].mean():.1f}",
            delta="",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="BMI (AD)",
            value=f"{ad['BMI'].mean():.1f}",
            delta="",
            card_type="success"
        ), unsafe_allow_html=True
    )


st.markdown("---")
st.markdown(
    Components.page_header(
        "📉 Statistical Tests"
    ), unsafe_allow_html=True
)
col1, col2 = st.columns(2, border=True)
with col1:
    st.subheader(":orange[T‑Tests (Continuous Variables)]", divider="orange")
    continuous_vars = [
        "Age", "BMI", "MMSE", "FunctionalAssessment", "AlcoholConsumption", 
        "PhysicalActivity", "DietQuality", "SleepQuality"
    ]
    for var in continuous_vars:
        g0 = df[df['Diagnosis'] ==0][var]
        g1 = df[df['Diagnosis'] ==1][var]
        t, p = ttest_ind(g0, g1)
        st.write(f"**{var}** — p = {p:.4f}")

with col2:
    st.subheader(":red[Chi-Square Tests (Categorical)]", divider="red")
    categorical_vars = [
        "Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
        "Diabetes", "Depression", "Hypertension"
    ]
    for var in categorical_vars:
        table = pd.crosstab(df[var], df['Diagnosis'])
        chi2, p, dof, exp = chi2_contingency(table)
        st.write(f"**{var}** — p = {p:.4f}")

significant_t = sum([
    ttest_ind(df[df['Diagnosis']==0][var], df[df['Diagnosis'] == 1][var])[1] < 0.05
    for var in continuous_vars
])

significant_chi = sum([
    chi2_contingency(pd.crosstab(df[var], df['Diagnosis']))[1] < 0.05
    for var in categorical_vars
])
symptom_cols = [ 
        'MemoryComplaints','BehavioralProblems','Confusion','Disorientation', 
        'PersonalityChanges','DifficultyCompletingTasks','Forgetfulness' 
] 
significant_symptoms = sum([
    mannwhitneyu(df[df['Diagnosis'] == 0][sym], df[df['Diagnosis'] == 1][sym])[1] < 0.05
    for sym in symptom_cols
])
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.metric_card(
            title="Significant T-Tests",
            value=f"{significant_t}",
            delta="🔬",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Significant Chi-Square Tests",
            value=f"{significant_chi}",
            delta="💊",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Significant Symptom Differences",
            value=f"{significant_symptoms}",
            delta="🩺",
            card_type="success"
        ), unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    Components.page_header(
        "🔮 Predictive Modeling"
    ), unsafe_allow_html=True
)

feature_columns = [
    col for col in df.columns if col not in [
        "PatientID", "Diagnosis", "DoctorInCharge", "Diagnosis_Label",
        "Gender_Label", "Ethnicity_Label", "Education_Label",
        "BP_Category", "BMI_Category", "Age_Group", "Cognitive_Status"
    ]
]
X = df[feature_columns].select_dtypes(include=[np.number]) 
y = df["Diagnosis"] 

X = X.fillna(X.median()) 

X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42, stratify=y 
)

scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

models = {} 
results = [] 

# Logistic Regression 
lr = LogisticRegression(max_iter=1000) 
lr.fit(X_train_scaled, y_train) 
y_pred_lr = lr.predict(X_test_scaled) 
y_proba_lr = lr.predict_proba(X_test_scaled)[:,1] 

results.append(["Logistic Regression", 
                lr.score(X_test_scaled,y_test), 
                roc_auc_score(y_test,y_proba_lr), 
                f1_score(y_test,y_pred_lr)]) 

lr_coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_[0],
    "Abs_Coefficient": np.abs(lr.coef_[0])
}).sort_values("Abs_Coefficient", ascending=False)
               
# Random Forest 
rf = RandomForestClassifier(n_estimators=100) 
rf.fit(X_train, y_train) 
y_pred_rf = rf.predict(X_test) 
y_proba_rf = rf.predict_proba(X_test)[:,1] 

fi = pd.DataFrame({ 
    "Feature": X.columns, 
    "Importance": rf.feature_importances_ 
}).sort_values("Importance", ascending=False)

results.append(["Random Forest", 
                rf.score(X_test,y_test), 
                roc_auc_score(y_test,y_proba_rf), 
                f1_score(y_test,y_pred_rf)]) 
                
# Gradient Boosting 
gb = GradientBoostingClassifier() 
gb.fit(X_train, y_train) 
y_pred_gb = gb.predict(X_test) 
y_proba_gb = gb.predict_proba(X_test)[:,1] 

results.append(["Gradient Boosting", 
                gb.score(X_test,y_test), 
                roc_auc_score(y_test,y_proba_gb), 
                f1_score(y_test,y_pred_gb)]) 
                
# SVM 
svm = SVC(kernel="rbf", probability=True) 
svm.fit(X_train_scaled, y_train) 
y_pred_svm = svm.predict(X_test_scaled) 
y_proba_svm = svm.predict_proba(X_test_scaled)[:,1] 

results.append(["SVM", 
                svm.score(X_test_scaled,y_test), 
                roc_auc_score(y_test,y_proba_svm), 
                f1_score(y_test,y_pred_svm)]) 
                
results_df = pd.DataFrame(results, columns=["Model","Accuracy","AUC","F1"]) 
st.dataframe(results_df) 

# ROC curves 
fig = go.Figure() 
preds = { 
    "Logistic Regression": y_proba_lr, 
    "Random Forest": y_proba_rf, 
    "Gradient Boosting": y_proba_gb, 
    "SVM": y_proba_svm 
} 

for name, proba in preds.items(): 
    fpr, tpr, _ = roc_curve(y_test, proba) 
    auc = roc_auc_score(y_test, proba) 
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})")) 
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random")) 
    st.plotly_chart(fig, width="stretch") 
    
# Feature importance (RF) 
st.subheader(":violet[Top Features (Random Forest)]", divider="violet") 
fi = pd.DataFrame({ 
    "Feature": X.columns, 
    "Importance": rf.feature_importances_ 
}).sort_values("Importance", ascending=False) 

st.dataframe(fi.head(20))

st.markdown("---")
st.markdown(
    Components.page_header(
        "🩺 Key Insights"
    ), unsafe_allow_html=True
)
best = results_df.loc[results_df["AUC"].idxmax()]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Best Model",
            value=f"{best["Model"]}",
            delta="🧮",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Best AUC",
            value=f"{best['AUC']*100:.1f}%",
            delta="📑",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Best Accuracy",
            value=f"{best['Accuracy']*100:.1f}%",
            delta="🧪",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Best F1‑Score",
            value=f"{best['F1']*100:.1f}%",
            delta="🧫",
            card_type="error"
        ), unsafe_allow_html=True
    )
top_risk = fi.iloc[0] 
top_protective = lr_coef[lr_coef["Coefficient"] < 0].iloc[0]
st.divider()
col1, col2, col3, col4 = st.columns(4, border=True)
with col1:
    st.markdown(
        Components.insight_box(
            "🔴 TOP 5 RISK FACTORS (from feature importance):",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Functional Assessment:</strong> Importance: 0.1792</li>
                <li><strong>ADL:</strong> Importance: 0.1623</li>
                <li><strong>MMSE:</strong> Importance: 0.1166</li>
                <li><strong>Memory Complaints:</strong> Importance: 0.0831</li>
                <li><strong>Behavioral Problems:</strong> Importance: 0.0413</li>
            </ul>
            """,
            "warning"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.insight_box(
            "🟢 TOP 5 PROTECTIVE FACTORS (from logistic regression):",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Functional Assessment:</strong> Coefficient: -1.3180</li>
                <li><strong>ADL:</strong> Coefficient: -1.2692</li>
                <li><strong>MMSE:</strong> Coefficient: -0.8571</li>
                <li><strong>Personality Changes:</strong> Coefficient: -0.2000</li>
                <li><strong>Forgetfulness:</strong> Coefficient: -0.1911</li>
            </ul>
            """,
            "success"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.insight_box(
            "👥 DEMOGRAPHIC INSIGHTS:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Average age (No AD):</strong> 74.9 ± 8.9</li>
                <li><strong>Average age (AD):</strong> 74.8 ± 9.1</li>
                <li><strong>AD Rate in Males:</strong> 36.4%</li>
                <li><strong>AD Rate in Females:</strong> 34.4%</li>
            </ul>
            """,
            "info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.insight_box(
            "🧠 COGNITIVE MARKERS:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Avg MMSE (No AD):</strong> 16.27</li>
                <li><strong>Avg MMSE (AD):</strong> 11.99</li>
                <li><strong>Difference:</strong> 4.27 points</li>
            </ul>
            """,
            "info"
        ), unsafe_allow_html=True
    )

st.markdown("---")
col1, col2, col3 = st.columns(3, border=True)
with col1:
    st.markdown(
        Components.insight_box(
            "🏃 LIFESTYLE IMPACT:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Avg Lifestyle Risk Score (No AD):</strong> 2.50</li>
                <li><strong>Avg Lifestyle Risk Score (AD):</strong> 2.54</li>
                <li><strong>Higher scores indicate worse lifestyle habits</strong></li>
            </ul>
            """,
            "warning"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.insight_box(
            "⚠️ SYMPTOM PREVALENCE IN AD PATIENTS:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Memory Complaints:</strong> 37.6%</li>
                <li><strong>Forgetfulness:</strong> 30.1%</li>
                <li><strong>Behavioral Problems:</strong> 26.7%</li>
                <li><strong>Confusion:</strong> 19.5%</li>
                <li><strong>Difficulty Completing Tasks:</strong> 16.3%</li>
            </ul>
            """,
            "info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.insight_box(
            "🏥 MEDICAL COMORBIDITIES IN AD PATIENTS:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>Hypertension:</strong> 16.6% (AD) vs 14.0% (No AD)</li>
            </ul>
            """,
            "info"
        ), unsafe_allow_html=True
    )
st.markdown("---")
st.markdown(
    Components.page_header(
        "💡 Clinical Recommendations"
    ), unsafe_allow_html=True
)
st.divider()
with st.container(border=True):
    st.markdown(
        Components.insight_box(
            "📋 RISK ASSESSMENT PROTOCOL:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <ol>
                <li value="1"><strong>Prioritize MMSE screening for patients over 65 with: </strong>
                </ol>
                    <ul>
                        <li>Low functional assessment scores</li>
                        <li>Multiple cognitive symptoms (≥3)</li>
                        <li>Family history of Alzheimer's </li>
                <br>
                <ol>
                <li value="2"><strong>Monitor high-risk indicators:</strong>
                </ol>
                    <ul>
                        <li>MMSE scores below 24 (mild cognitive impairment threshold)</li>
                        <li>Functional assessment scores below 5 </li>
                        <li>Presence of memory complaints + disorientation</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ul>
    """,
        "error"
    ), unsafe_allow_html=True
    )
st.divider()
with st.container(border=True):
    st.markdown(
        Components.insight_box(
            "🎯 PREVENTIVE INTERVENTIONS:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <ul>
                <li value="1"><strong>Lifestyle Modifications (Modifiable Risk Factors):</strong></li>
                </ul>
                <li>- Increase physical activity (target: >7 hrs/week) </li>
                <li>- Improve diet quality (Mediterranean diet recommended) </li>
                <li>- Optimize sleep quality (7-9 hours/night) </li>
                <li>- Reduce alcohol consumption (<14 drinks/week)</li>
                <li>- Smoking cessation programs  </li>
                <br>
                <ul>
                <li value="2"><strong>Medical Management:</strong></li>
                </ul>
                <li>- Control hypertension (target: <130/80 mmHg) </li>
                <li>- Manage cholesterol levels</li>
                <li>- Screen and treat depression early</li>
                <li>- Monitor and control diabetes</li>
            </ul>
            """,
            "success"
        ), unsafe_allow_html=True
    )
st.divider()
with st.container(border=True):
    st.markdown(
        Components.insight_box(
            "🔬 EARLY DETECTION STRATEGY:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <ul>
                <li value="1"><strong>Regular cognitive screening for at-risk populations</strong></li>
                </ul>
                <li>- Age 65+ with family history</li>
                <li>- Patients with multiple comorbidities</li>
                <li>- Those reporting subjective cognitive decline</li>
                <br>
                <ul>
                <li value="2"><strong>Use predictive model for risk stratification:</strong></li>
                </ul>
                <li><strong>High risk:</strong>Model probability >0.7 </li>
                <li><strong>Moderate risk:</strong>Model probability 0.4-0.7 </li>
                <li><strong>Low risk:</strong>Model probability <0.4 </li>
            </ul>
            """,
            "success"
        ), unsafe_allow_html=True
    )
st.divider()
with st.container(border=True):
    st.markdown(
        Components.insight_box(
            "### 📊 MONITORING & FOLLOW-UP:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li value="1"><strong>High-risk patients:</strong> Quarterly assessments</li>
                <li value="2"><strong>Moderate-risk:</strong> Bi-annual assessments</li>
                <li value="3"><strong>Track progression using:</strong></li>
                <ul>
                <li>- MMSE scores (change >3 points = significant)</li>
                <li>- Functional assessment</li>
                <li>- ADL capacity</li>
                <li>- Symptom emergence</li>
                </ul>
            </ul>
            """,
            "warning"
        ), unsafe_allow_html=True
    )
st.divider()
with st.container(border=True):
    st.markdown(
        Components.insight_box(
            "### 🏥 HEALTHCARE SYSTEM INTEGRATION:",
            """
            <ul style="margin: 0; padding-left: 20px;">
                <li value="1">Deploy predictive model in electronic health records</li>
                <li value="2">Create automated alerts for high-risk patients</li>
                <li value="3">Establish multidisciplinary care teams</li>
                <li value="4">Implement patient education programs</li>
            </ul>
            """,
            "info"
        ), unsafe_allow_html=True
    )

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
