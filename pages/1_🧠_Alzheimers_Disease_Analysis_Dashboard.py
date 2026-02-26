import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# Title
st.markdown(
    Components.page_header(
        "🧠 Alzheimers Disease Analysis Dashboard"
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
