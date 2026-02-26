import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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


# Title
st.markdown(
    Components.page_header(
        "🌡️ Weather NYC Analysis Dashboard"
    ), unsafe_allow_html=True
)
