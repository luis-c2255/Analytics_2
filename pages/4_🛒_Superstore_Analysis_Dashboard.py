import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Superstore Analysis Dashboard", "🛒")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('global_superstore_2016.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range filter
min_date = df['Order Date'].min()
max_date = df['Order Date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value = (min_date_max_date),
    min_value=min_date,
    max_value=max_date
)
# Region filter
regions = ['All'] + sorted(df['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)

# Category filter
categories = ['All'] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Select Category", categories)

# Segment filter
segments = ['All'] + sorted(df['Segment'].unique().tolist())
selected_segment = st.sidebar.selectbox("Select Segment", segments)

# Apply filters
df_filtered = df.copy()
df_filtered = df_filtered[
    (df_filtered['Order Date'] >= pd.to_datetime(date_range[0])) &
    (df_filtered['Order Date'] <= pd.to_datetime(date_range[1]))
]

if selected_region != 'All':
    df_filtered = df_filtered[df_filtered['Region'] == selected_region]

if selected_category != 'All':
    df_filtered = df_filtered[df_filtered['Category'] == selected_category]

if selected_segment != 'All':
    df_filtered = df_filtered[df_filtered['Segment'] == selected_segment]

# Title
st.markdown(
    Components.page_header(
        "🛒 Superstore Sales & Profitability Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("   ")

# ===== KEY METRICS =====
st.subheader("💼 :red[Key Business Metrics]", divider="red")

col1, col2, col3, col4, col5 = st.columns(5)

total_sales = df_filtered['Sales'].sum()
total_profit = df_filtered['Profit'].sum()
total_orders = df_filtered['Order ID'].nunique()
avg_profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
total_customers = df_filtered['Customer ID'].nunique()

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Sales",
            value=f"${total_sales:,.0f}",
            delta="💰",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Total Profit",
            value=f"${total_profit:,.0f}",
            delta="💵",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Total Orders",
            value=f"{total_orders:,}",
            delta="📦",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Profit Margin",
            value=f"{avg_profit_margin:.2f}%",
            delta="📈",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col5:
    st.markdown(
        Components.metric_card(
            title="Customers",
            value=f"{total_customers:,}",
            delta="👥",
            card_type="info"
        ), unsafe_allow_html=True
    )
    
st.markdown("   ")
st.subheader("🌍 :blue[Regional & Category Performance]", divider="blue")

st.markdown("   ")
st.subheader("Sales by Region", divider="blue")
regional_sales = df_filtered.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

fig_region = px.bar(
    regional_sales,
    x='Region',
    y='Sales',
    color='Profit',
    title='Sales and Profit by Region',
    color_continuous_scale='RdYlGn',
    text='Sales'
)
fig_region.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
fig_region.update_layout(height=400)
st.plotly_chart(fig_region, width="stretch")
st.markdown("   ")
st.subheader(":green[Sales by Category]", divider="green")
category_sales = df_filtered.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

fig_category = px.pie(
    category_sales,
    values='Sales',
    names='Category',
    title='Sales Distribution by Category',
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig_category.update_traces(textposition='inside', textinfo='percent+label')
fig_category.update_layout(height=400)
st.plotly_chart(fig_category, width="stretch")

st.markdown("   ")
st.subheader("🏆 :yellow[Top Performers & ⚠️ Underperformers]", divider="yellow")

st.markdown("   ")
st.subheader(":orange[Top 10 Countries by Profit]")
country_profit = df_filtered.groupby('Country').agg({
    'Profit': 'sum',
    'Sales': 'sum'
}).reset_index()

country_profit['Profit Margin %'] = (country_profit['Profit'] / country_profit['Sales']) * 100
top_countries = country_profit.nlargest(10, 'Profit')

fig_top_countries = px.bar(
    top_countries,
    x='Profit',
    y='Country',
    orientation='h',
    title='Top 10 Most Profitable Countries',
    color='Profit Margin %',
    color_continuous_scale='Greens',
    text='Profit'
)
fig_top_countries.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
fig_top_countries.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_top_countries, width="stretch")

st.markdown("   ")
st.subheader(":yellow[Bottom 10 Countries by Profit]")
bottom_countries = country_profit.nsmallest(10, 'Profit')

fig_bottom_countries = px.bar(
    bottom_countries,
    x='Profit',
    y='Country',
    orientation='h',
    title='Bottom 10 Countries (Loss-Making)',
    color='Profit',
    color_continuous_scale='Reds',
    text='Profit'
)
fig_bottom_countries.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
fig_bottom_countries.update_layout(height=400, yaxis={'categoryorder':'total descending'})
st.plotly_chart(fig_bottom_countries, width="stretch")

st.markdown("   ")
st.subheader("📦 :orange[Sub-Category Profitability Analysis]", divider="orange")
subcategory_performance = df_filtered.groupby('Sub-Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).reset_index()

subcategory_performance['Profit Margin %'] = (subcategory_performance['Profit'] / subcategory_performance['Sales']) * 100
subcategory_performance = subcategory_performance.sort_values('Profit', ascending=True)

fig_subcategory = px.bar(
    subcategory_performance,
    x='Profit',
    y='Sub-Category',
    orientation='h',
    title='Profit by Sub-Category (Sorted by Profitability)',
    color='Profit',
    color_continuous_scale='RdYlGn',
    text='Profit'
)
fig_subcategory.update_traces(texttemplate='$%{text:.2s}', textposition='outside')
fig_subcategory.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_subcategory, width="stretch")

st.markdown("   ")
st.subheader("📅 :blue[Time-Based Sales & Profit Trends]", divider="blue")

# Monthly trends
monthly_data = df_filtered.groupby(df_filtered['Order Date'].dt.to_period('M')).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

monthly_data['Order Date'] = monthly_data['Order Date'].dt.to_timestamp()

fig_trends = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Monthly Sales Trend', 'Monthly Profit Trend'),
    vertical_spacing=0.15
)

fig_trends.add_trace(
    go.Scatter(
        x=monthly_data['Order Date'],
        y=monthly_data['Sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy'
    ),
    row=1, col=1
)

fig_trends.add_trace(
    go.Scatter(
        x=monthly_data['Order Date'],
        y=monthly_data['Profit'],
        mode='lines+markers',
        name='Profit',
        line=dict(color='#2ca02c', width=3),
        fill='tozeroy'
    ),
    row=2, col=1
)

fig_trends.update_xaxes(title_text="Date", row=2, col=1)
fig_trends.update_yaxes(title_text="Sales ($)", row=1, col=1)
fig_trends.update_yaxes(title_text="Profit ($)", row=2, col=1)
fig_trends.update_layout(height=600, showlegend=False)

st.plotly_chart(fig_trends, width="stretch")

st.markdown("   ")
st.subheader("👥 :violet[Customer Segment & 🚚 Shipping Analysis]", divider="violet")

st.subheader(":rainbow[Segment Performance]")
segment_data = df_filtered.groupby('Segment').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'nunique'
}).reset_index()

fig_segment = go.Figure()
fig_segment.add_trace(go.Bar(
    name='Sales',
    x=segment_data['Segment'],
    y=segment_data['Sales'],
    marker_color='lightskyblue'
))
fig_segment.add_trace(go.Bar(
    name='Profit',
    x=segment_data['Segment'],
    y=segment_data['Profit'],
    marker_color='lightgreen'
))
fig_segment.update_layout(
    title='Sales vs Profit by Customer Segment',
    barmode='group',
    height=400
)
st.plotly_chart(fig_segment, width="stretch")

st.subheader(":violet[Shipping Mode Analysis]")
shipping_data = df_filtered.groupby('Ship Mode').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'nunique'
}).reset_index()
shipping_data['Profit Margin %'] = (shipping_data['Profit'] / shipping_data['Sales']) * 100

fig_shipping = px.scatter(
    shipping_data,
    x='Sales',
    y='Profit',
    size='Order ID',
    color='Profit Margin %',
    text='Ship Mode',
    title='Shipping Mode: Sales vs Profit',
    color_continuous_scale='Viridis',
    size_max=60
)
fig_shipping.update_traces(textposition='top center')
fig_shipping.update_layout(height=400)
st.plotly_chart(fig_shipping, width="stretch")

st.markdown("   ")
st.subheader("🔥 :orange[Profitability Heatmap: Region × Category]", divider="orange")

heatmap_data = df_filtered.pivot_table(
    values='Profit',
    index='Region',
    columns='Category',
    aggfunc='sum'
)

fig_heatmap = px.imshow(
    heatmap_data,
    labels=dict(x="Category", y="Region", color="Profit ($)"),
    title='Profit Heatmap by Region and Category',
    color_continuous_scale='RdYlGn',
    aspect='auto',
    text_auto='.2s'
)
fig_heatmap.update_layout(height=500)
st.plotly_chart(fig_heatmap, width="stretch")

st.markdown("   ")
st.subheader("📋 :green[Detailed Performance Data]", divider="green")
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Countries", "📦 Sub-Categories", "🏷️ Products", "👤 Customers"])

with tab1:
    st.subheader(":blue[Country-Level Performance]")
    country_detail = df_filtered.groupby('Country').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
country_detail['Profit Margin %'] = (country_detail['Profit'] / country_detail['Sales']) * 100
country_detail = country_detail.sort_values('Sales', ascending=False)
country_detail.columns = ['Country', 'Sales ($)', 'Profit ($)', 'Orders', 'Quantity', 'Profit Margin (%)']

st.dataframe(
    country_detail.style.format({
        'Sales ($)': '${:,.2f}',
        'Profit ($)': '${:,.2f}',
        'Orders': '{:,}',
        'Quantity': '{:,}',
        'Profit Margin (%)': '{:.2f}%'
    }).background_gradient(subset=['Profit ($)'], cmap='RdYlGn'),
    width="stretch",
    height=400
)
with tab2:
    st.subheader(":orange[Sub-Category Performance]")
    subcat_detail = df_filtered.groupby('Sub-Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
subcat_detail['Profit Margin %'] = (subcat_detail['Profit'] / subcat_detail['Sales']) * 100
subcat_detail = subcat_detail.sort_values('Profit', ascending=False)
subcat_detail.columns = ['Sub-Category', 'Sales ($)', 'Profit ($)', 'Orders', 'Quantity', 'Profit Margin (%)']

st.dataframe(
    subcat_detail.style.format({
        'Sales ($)': '${:,.2f}',
        'Profit ($)': '${:,.2f}',
        'Orders': '{:,}',
        'Quantity': '{:,}',
        'Profit Margin (%)': '{:.2f}%'
    }).background_gradient(subset=['Profit ($)'], cmap='RdYlGn'),
    width="stretch",
    height=400
)

with tab3:
    st.subheader(":yellow[Product Performance (Top 100 by Sales)]")
    product_detail = df_filtered.groupby('Product Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
product_detail['Profit Margin %'] = (product_detail['Profit'] / product_detail['Sales']) * 100
product_detail = product_detail.sort_values('Sales', ascending=False).head(100)
product_detail.columns = ['Product Name', 'Sales ($)', 'Profit ($)', 'Orders', 'Quantity', 'Profit Margin (%)']

st.dataframe(
    product_detail.style.format({
        'Sales ($)': '${:,.2f}',
        'Profit ($)': '${:,.2f}',
        'Orders': '{:,}',
        'Quantity': '{:,}',
        'Profit Margin (%)': '{:.2f}%'
    }).background_gradient(subset=['Profit ($)'], cmap='RdYlGn'),
    width="stretch", height=400
)

with tab4:
    st.subheader(":violet[Top 50 Customers by Sales]")
    customer_detail = df_filtered.groupby('Customer Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
customer_detail['Avg Order Value'] = customer_detail['Sales'] / customer_detail['Order ID']
customer_detail['Profit Margin %'] = (customer_detail['Profit'] / customer_detail['Sales']) * 100
customer_detail = customer_detail.sort_values('Sales', ascending=False).head(50)
customer_detail.columns = ['Customer Name', 'Sales ($)', 'Profit ($)', 'Orders', 'Quantity', 'Avg Order Value ($)', 'Profit Margin (%)']

st.dataframe(
    customer_detail.style.format({
        'Sales ($)': '${:,.2f}',
        'Profit ($)': '${:,.2f}',
        'Orders': '{:,}',
        'Quantity': '{:,}',
        'Avg Order Value ($)': '${:,.2f}',
        'Profit Margin (%)': '{:.2f}%'
    }).background_gradient(subset=['Sales ($)'], cmap='Blues'),
    width="stretch",
    height=400
)
# ===== FOOTER WITH DOWNLOAD OPTIONS =====
st.markdown("   ")
st.subheader("💾 :violet[Export Data]", divider="violet")

col1, col2, col3 = st.columns(3)

with col1:
    # Download filtered data
    csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_filtered,
        file_name='filtered_superstore_data.csv',
        mime='text/csv'
    )

with col2:
    # Download country summary
    country_summary_csv = country_detail.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Country Summary (CSV)",
        data=country_summary_csv,
        file_name='country_performance_summary.csv',
        mime='text/csv'
    )

with col3:
    # Download product summary
    product_summary_csv = product_detail.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Product Summary (CSV)",
        data=product_summary_csv,
        file_name='product_performance_summary.csv',
        mime='text/csv'
    )

st.markdown("   ")
st.subheader("💡 :yellow[Quick Insights]", divider="yellow")

# Calculate key insights
most_profitable_region = regional_sales.loc[regional_sales['Profit'].idxmax(), 'Region']
least_profitable_region = regional_sales.loc[regional_sales['Profit'].idxmin(), 'Region']
best_category = category_sales.loc[category_sales['Profit'].idxmax(), 'Category']

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        Components.metric_card(
            title="Most Profitable Region",
            value=f"{most_profitable_region}",
            delta="🏆",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Least Profitable Region",
            value=f"{least_profitable_region}",
            delta="⚠️",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Best Category",
            value=f"{best_category}",
            delta="📦",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col4:
    if len(unprofitable) > 0:
        loss_percentage = (abs(unprofitable['Profit'].sum()) / total_profit * 100)
    st.markdown(
        Components.metric_card(
            title="Losses represent",
            value=f"{loss_percentage:.1f}% of total profit",
            delta="🚨",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader("💎 :blue[Key Insights & Actionable Recommendations]", divider="blue")
# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>🛒 Superstore Analysis Dashboard</strong></p>
    <p>Analysis of a dataset from a Superstore sales</p>
    <p style='font-size: 0.9rem;'>Navigate using the sidebar to explore different datasets</p>
</div>
""", unsafe_allow_html=True)
