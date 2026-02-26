import streamlit as st
from utils.theme import Components, Colors, apply_chart_theme, init_page


st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    Components.page_header("📊 Multiple Analysis Dashboard"), unsafe_allow_html=True)

with st.container(height="content", width="stretch", horizontal_alignment="center"):    
    st.image("utils/image.svg")

col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("Alzheimers Disease Analysis Dashboard", 
    "https://analytics2.streamlit.app/Alzheimers_Disease_Analysis_Dashboard",
    icon="🧠", icon_position="left", width="stretch"
    )

with col2:
    st.link_button("Weather NYC Analysis Dashboard", 
    "https://analytics2.streamlit.app/Weather_NYC_Analysis_Dashboard",
    icon="🌡", icon_position="left", width="stretch"
    )
with col3:
    st.link_button("Yahoo Stock Analysis Dashboard", 
    "https://analytics2.streamlit.app/Yahoo_Stock_Analysis_Dashboard", 
    icon="🆙", icon_position="left", width="stretch"
    )
st.markdown("---")
col4, col5 = st.columns(2)

with col4:
    st.link_button("Superstore Analysis Dashboard", 
    "https://analytics2.streamlit.app/Superstore_Analysis_Dashboard", 
    icon="🛒", icon_position="left", width="stretch"
    )
with col5:
    st.link_button("Telco Customer Churn Analysis Dashboard", 
    "https://analytics2.streamlit.app/Telco_Customer_Churn_Analysis_Dashboard", 
    icon="👥", icon_position="left", width="stretch"
    )



# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>📊 Multiple Analysis Dashboard</strong></p>
    <p>Multiple Dashboards from several datasets analyzed</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
