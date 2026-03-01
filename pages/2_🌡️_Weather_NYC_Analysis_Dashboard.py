import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Weather NYC Analysis Dashboard", "🌡️")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('weather_data_nyc.csv')
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df = df.sort_values('date').reset_index(drop=True)
    cols = ['precipitation', 'snow fall', 'snow depth']
    df[cols] = (df[cols]
    .replace("T", 0.005)
    .astype(float)
)

    # Feature engineering
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day_of_week'] = df['date'].dt.day_name()
    df['temp_range'] = df['maximum temperature'] - df['minimum temperature']
    df['is_rainy'] = df['precipitation'] > 0
    df['is_snowy'] = df['snow fall'] > 0
    return df
df = load_data()


# Title
st.markdown(
    Components.page_header(
        "🌡️ Weather NYC Analysis Dashboard"
    ), unsafe_allow_html=True
)
# Sidebar filters
st.sidebar.header("🔧 Filters & Controls")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Month filter
selected_months = st.sidebar.multiselect(
    "Select Months",
    options=df['month_name'].unique(),
    default=df['month_name'].unique()
)

# Temperature threshold
temp_threshold = st.sidebar.slider(
    "Temperature Threshold (ºF)",
    int(df['minimum temperature'].min()),
    int(df['maximum temperature'].max()),
    int(df['average temperature'].mean())
)

# Apply filters
if len(date_range) == 2:
    mask = (df['date'] >=pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
    filtered_df = df[mask]
else:
    filtered_df = df
if selected_months:
    filtered_df = filtered_df[filtered_df['month_name'].isin(selected_months)]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        Components.metric_card(
            title="🌡️ Avg Temperature",
            value=f"{filtered_df['average temperature'].mean():.1f}°F",
            delta=f"{filtered_df['average temperature'].mean() - df['average temperature'].mean():.1f}°F vs Overall",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="🔥 Hottest Day",
            value=f"{filtered_df['maximum temperature'].max()}°F",
            delta=f"{filtered_df.loc[filtered_df['maximum temperature'].idxmax(), 'date'].strftime('%b %d')}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="❄️ Coldest Day",
            value=f"{filtered_df['minimum temperature'].min()}°F",
            delta=f"{filtered_df.loc[filtered_df['minimum temperature'].idxmin(), 'date'].strftime('%b %d')}",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    rainy_days = filtered_df['is_rainy'].sum()
    st.markdown(
        Components.metric_card(
            title="🌧️ Rainy Days",
            value=f"{rainy_days}",
            delta=f"{rainy_days/len(filtered_df)*100:.1f}% of period",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col5:
    st.markdown(
        Components.metric_card(
            title="🌨️ Total Snowfall",
            value=f"{filtered_df['snow fall'].sum():.1f}",
            delta=f"{filtered_df['is_snowy'].sum()} snowy days",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.markdown(
    Components.page_header(
        "📈 Temperature Trends"
    ), unsafe_allow_html=True
)
st.subheader(":blue[Temperature Evolution Over Time]", divider="blue")
# Interactive line chart
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(
    x=filtered_df['date'],
    y=filtered_df['maximum temperature'],
    name='Max Temperature',
    line=dict(color='red', width=2),
    hovertemplate='Max Temp: %{y}°FDate: %{x}'
))

fig_temp.add_trace(go.Scatter(
    x=filtered_df['date'],
    y=filtered_df['minimum temperature'],
    name='Min Temperature',
    line=dict(color='blue', width=2),
    hovertemplate='Min Temp: %{y}°FDate: %{x}'))

fig_temp.add_trace(go.Scatter( 
    x=filtered_df['date'],
    y=filtered_df['average temperature'],
    name='Avg Temperature',
    line=dict(color='green', width=3),
    fill='tonexty',
    hovertemplate='Avg Temp: %{y}°FDate: %{x}'
))
# Add threshold line
fig_temp.add_hline(
    y=temp_threshold,
    line_dash="dash",
    line_color="orange",
    annotation_text=f"Threshold: {temp_threshold}°F"
)
fig_temp.update_layout(
    title="Daily Temperature Trends",
    xaxis_title="Date",
    yaxis_title="Temperature (°F)",
    hovermode='x unified',
    height=500,
    template='plotly_white'
)
st.plotly_chart(fig_temp, width="stretch")







st.markdown(
    Components.page_header(
        "🌧️ Precipitation Analysis"
    ), unsafe_allow_html=True
)







st.markdown(
    Components.page_header(
        "❄️ Snow Patterns"
    ), unsafe_allow_html=True
)








st.markdown(
    Components.page_header(
        "📊 Statistical Insights"
    ), unsafe_allow_html=True
)






st.markdown(
    Components.page_header(
        "🔍 Custom Explorer"
    ), unsafe_allow_html=True
)
# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🌡️ Weather NYC Analysis Dashboar</strong></p>
    <p>Analysis of New York City weather dataset</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
