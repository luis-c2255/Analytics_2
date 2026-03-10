import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Serial Killers Analysis Dashboard", "🗡️")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('serial_killers.csv')

# Data Cleaning
df_clean = df.copy()

# Parse dates
df_clean['Date Apprehended'] = pd.to_datetime(df_clean['Date Apprehended'], errors='coerce')
df_clean['Born Date'] = pd.to_datetime(df_clean['Born Date'], errors='coerce')
df_clean['Died Date'] = pd.to_datetime(df_clean['Died Date'], errors='coerce')

# Calculate derived fields
df_clean['Active_Years'] = df_clean['End year'] - df_clean['Start year']
df_clean['Active_Years'] = df_clean['Active_Years'].apply(lambda x: max(x, 0) if pd.notna(x) else np.nan)
df_clean['Decade'] = (df_clean['Start year'] // 10 * 10).astype('Int64')

# Victim categorization
def categorize_victims(count):
    if pd.isna(count):
        return 'Unknown'
    elif count < 5:
        return '2-4 victims'
    elif count < 10:
        return '5-9 victims'
    elif count < 20:
        return '10 - 19 victims'
    elif count < 50:
        return '20 - 49 victims'
    else:
        return '+50 victims'

df_clean['Victim_Category'] = df_clean['Proven victims'].apply(categorize_victims)

# Penalty categorization
def categorize_penalty(penalty):
    if pd.isna(penalty):
        return 'Unknown'
    penalty_lower = str(penalty).lower()
    if 'death' in penalty_lower or 'execution' in penalty_lower:
        return 'Death Penalty'
    elif 'life' in penalty_lower:
        return 'Life Imprisonment'
    elif 'years' in penalty_lower:
        return 'Fixed Term'
    else:
        return 'Other'
df_clean['Penalty_Category'] = df_clean['Criminal Penalty'].apply(categorize_penalty)

# Region mapping
region_map = {
    'United States': 'North America',
    'Canada': 'North America',
    'Mexico': 'North America',
    'United Kingdom': 'Europe',
    'Germany': 'Europe',
    'France': 'Europe',
    'Italy': 'Europe',
    'Russia': 'Europe',
    'Spain': 'Europe',
    'Poland': 'Europe',
    'Netherlands': 'Europe',
    'Belgium': 'Europe',
    'Australia': 'Oceania',
    'Brazil': 'South America',
    'Colombia': 'South America',
    'Argentina': 'South America',
    'India': 'Asia',
    'China': 'Asia',
    'Japan': 'Asia',
    'South Korea': 'Asia',
    'South Africa': 'Africa',
    }
df_clean['Region'] = df_clean['Country'].map(region_map).fillna('Other')

return df_clean

# Load data
df = load_data()

# Title
st.markdown(
    Components.page_header(
        "🗡️ Serial Killers Analytics Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("   ")
# Sidebar filters
st.sidebar.header("🎛️ Filters")

# Year range filter
min_year = int(df['Start year'].min()) if df['Start year'].notna().any() else 1900
max_year = int(df['Start year'].max()) if df['Start year'].notna().any() else 2020

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)
# Country filter
countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

# Victim category filter
victim_cats = ['All'] + sorted(df['Victim_Category'].unique().tolist())
selected_victim_cat = st.sidebar.selectbox("Select Victim Category", victim_cats)

# Region filter
regions = ['All'] + sorted(df['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)

# Apply filters
filtered_df = df.copy()
if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]
if selected_victim_cat != 'All':
    filtered_df = filtered_df[filtered_df['Victim_Category'] == selected_victim_cat]
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    filtered_df = filtered_df[
        (filtered_df['Start year'] >= year_range[0]) &
        (filtered_df['Start year'] <= year_range[1])
    ]

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Cases",
            value=f"{len(filtered_df):,}",
            delta="📊",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Total Proven Victims",
            value=f"{int(filtered_df['Proven victims'].sum()):,}",
            delta="💀",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Avg Victims/Case",
            value=f"{filtered_df['Proven victims'].mean():.1f}",
            delta="📈",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Countries",
            value=f"{filtered_df['Country'].nunique()}",
            delta="🌍",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col5:
    st.markdown(
        Components.metric_card(
            title="Avg Active Years",
            value=f"{filtered_df['Active_Years'].mean():.1f}",
            delta="⏱️",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader("🌍 :red[Geographic Analysis]", divider="blue")




st.markdown("   ")
st.subheader("📅 :green[Temporal Trends]", divider="green")





st.markdown("   ")
st.subheader("💀 :yellow[Victim Analysis]", divider="yellow")





st.markdown("   ")
st.subheader("⚖️ :orange[Criminal Justice]", divider="orange")





st.markdown("   ")
st.subheader("📋 :red[Data Explorer]", divider="red")