import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Yahoo Stock Analysis Dashboard", "🆙")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('yahoo_stock.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Feature engineering
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Price_Range'] = df['High'] - df['Low']
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Close_MA_50'] = df['Close'].rolling(window=50).mean()
    df['Close_MA_200'] = df['Close'].rolling(window=200).mean()
    df['Volatility_30D'] = df['Daily_Return'].rolling(window=30).std()
    df['Support_50D'] = df['Low'].rolling(window=50).min()
    df['Resistance_50D'] = df['High'].rolling(window=50).max()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df
df = load_data()

# Sidebar filters
st.sidebar.header("🔧 Filter Options")

# Date range selector
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value = min_date,
    max_value = max_date
)
# Filter dataframe by date
if len(date_range) == 2:
    mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
    filtered_df = df[mask].copy()
else:
    filtered_df = df.copy()

# Moving average selector
ma_options = st.sidebar.multiselect(
    "Select Moving Averages",
    options=['50-Day MA', '200-Day MA'],
    default=['50-Day MA', '200-Day MA']
)

# Chart type selector
chart_type = st.sidebar.selectbox(
    "Price Chart Type",
    options = ['Candlestick', 'Line Chart']
)
# Title
st.markdown(
    Components.page_header(
        "🆙 Yahoo Stock Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.sidebar.markdown("   ")
st.sidebar.markdown("---")
st.sidebar.markdown("📊 Dashboard shows comprehensive stock market analysis with technical indicators")

# Main metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    current_price = filtered_df['Close'].iloc[-1]
    st.markdown(
        Components.metric_card(
            title="Current Price",
            value=f"${current_price:,.2f}",
            delta="💲",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    total_return = ((filtered_df['Close'].iloc[-1] - filtered_df['Close'].iloc[0]) / filtered_df['Close'].iloc[0]) * 100
    st.markdown(
        Components.metric_card(
            title="Total Return",
            value=f"{total_return:.2f}%",
            delta="🔁",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    avg_volume = filtered_df['Volume'].mean()
    st.markdown(
        Components.metric_card(
            title="Avg Daily Volume",
            value=f"{avg_volume/1e9:.2f}B",
            delta="💱",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    volatility = filtered_df['Daily_Return'].std()
    st.markdown(
        Components.metric_card(
            title="Volatility (Std Dev)",
            value=f"{volatility:.2f}%",
            delta="💸",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
# Row 1: Price Chart and Volume
st.subheader("📊 :rainbow[Price Movement & Trading Volume]", divider="rainbow")



# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🆙 Yahoo Stock Analysis Dashboard</strong></p>
    <p>Analysis of Yahoo Stock dataset</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
