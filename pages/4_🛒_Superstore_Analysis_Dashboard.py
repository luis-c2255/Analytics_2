import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import openpyxl

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
    df = pd.read_excel('global_superstore_2016.xlsx')
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
    value = (min_date, max_date),
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
    height=600
)
st.plotly_chart(fig_segment, width="stretch")


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
fig_shipping.update_layout(height=600)
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

st.subheader("🌍 :blue[Country-Level Performance]")
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
    height=700
)
st.markdown("   ")
st.subheader("📦 :orange[Sub-Category Performance]")
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

st.markdown("   ")
st.subheader("🏷️ :yellow[Product Performance (Top 100 by Sales)]")
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

st.markdown("   ")
st.subheader("👤 :violet[Top 50 Customers by Sales]")
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
st.sidebar.subheader("💾 :yellow[Export Data]", divider="yellow")

csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_filtered,
        file_name='filtered_superstore_data.csv',
        mime='text/csv'
    )

country_summary_csv = country_detail.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
        label="📥 Download Country Summary (CSV)",
        data=country_summary_csv,
        file_name='country_performance_summary.csv',
        mime='text/csv'
    )


product_summary_csv = product_detail.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
        label="📥 Download Product Summary (CSV)",
        data=product_summary_csv,
        file_name='product_performance_summary.csv',
        mime='text/csv'
    )

st.markdown("   ")
st.subheader("🎯 :red[STRATEGIC INSIGHTS & RECOMMENDATIONS]", divider="red")
with st.expander("1️⃣ PROFITABILITY CONCERNS"):
    st.markdown("""
    - ⚠️ 2672 products are generating losses
    - 💸 Total losses: $920,646.16 (62.7% of total profit)
    ---
    **📊 Top 5 Loss-Making Sub-Categories:**
    1. Tables: $144,123.15
    2. Bookcases: $101,446.30
    3. Phones: $96,417.66
    4. Chairs: $96,084.90
    5. Machines: $78,672.74
    ---
    **✅ RECOMMENDATION:**
    - Investigate pricing strategy for loss-making products
    - Consider discontinuing products with consistent losses
    - Review supplier contracts and shipping costs for these items
    - Implement minimum order values or bundle unprofitable items
    """)

with st.expander("2️⃣ REGIONAL PERFORMANCE GAPS"):
    st.markdown("""
    **📈 Regional Profit Margins:**
    1. Canada: 26.62% ($17,817 profit)
    2. Eastern Europe: 24.86% ($77,085 profit)
    3. North Africa: 24.80% ($57,836 profit)
    ....
    21. Western Asia: -17.00% ($-53,922 profit)
    22. Western Africa: -28.99% ($-50,408 profit)
    23. Central Asia: -37.71% ($-7,282 profit)

    - ⚠️ 64.3% profit margin gap between best and worst regions
    ---
    **✅ RECOMMENDATION:**
    - Study best practices from Canada and apply to Central Asia
    - Analyze market-specific factors (competition, pricing, customer preferences)
    - Consider regional pricing strategies
    - Evaluate logistics and shipping costs by region
    """)

with st.expander("3️⃣ PRODUCT MIX OPTIMIZATION"):
    st.markdown("""
    **📦 Category Performance:**
    - Technology: 13.99% margin | $4,744,557 sales
    - Office Supplies: 13.69% margin | $3,787,493 sales
    - Furniture: 6.94% margin | $4,110,452 sales
    
    - ⚠️ 422 high-volume products have <10% profit margin
    ---
    **Top 5 High-Volume, Low-Margin Products:**
    1. Apple Smart Phone, Full Size... - 6.8% margin, $86,936 sales
    2. Office Star Executive Leather Armchair, Adjustable... - 9.3% margin, $50,662 sales
    3. Samsung Smart Phone, Cordless... - -0.4% margin, $48,653 sales
    4. Samsung Smart Phone, VoIP... - 8.6% margin, $45,406 sales
    5. Cisco Smart Phone, Cordless... - 9.8% margin, $41,022 sales
    ---
    **✅ RECOMMENDATION:**
    - Renegotiate supplier prices for high-volume, low-margin items
    - Increase prices strategically on popular products with low margins
    - Create product bundles to improve overall margin
    - Focus marketing on high-margin categories
    """)

with st.expander("4️⃣ CUSTOMER SEGMENT PROFITABILITY"):
    st.markdown("""
    **👥 Segment Analysis:**
     Consumer:
     - Profit Margin: 11.51%
     - Customers: 8,987
     - Avg Profit/Customer: $83.37
     - Avg Sales/Customer: $724.15

    Corporate:
    - Profit Margin: 11.54%
    - Customers: 5,221
    - Avg Profit/Customer: $84.51
    - Avg Sales/Customer: $732.56

    Home Office:
    - Profit Margin: 11.99%
    - Customers: 3,207
    - Avg Profit/Customer: $86.38
    - Avg Sales/Customer: $720.25
    ---
    **✅ RECOMMENDATION:**
    - Focus acquisition efforts on Home Office segment (highest margin)
    - Develop loyalty programs tailored to each segment
    - Create segment-specific product bundles
    - Analyze customer lifetime value (CLV) by segment
    """)

with st.expander("5️⃣ SHIPPING & LOGISTICS OPTIMIZATION"):
    st.markdown("""
    **🚚 Shipping Mode Analysis:**
     First Class:
     - Orders: 3,845
     - Profit Margin: 11.37%
     - Avg Shipping Cost: $80.26
     - Shipping as % of Sales: 16.85%

    Same Day:
    - Orders: 1,349
    - Profit Margin: 11.42%
    - Avg Shipping Cost: $86.09
    - Shipping as % of Sales: 17.41%

    Second Class:
    - Orders: 5,146
    - Profit Margin: 11.40%
    - Avg Shipping Cost: $61.21
    - Shipping as % of Sales: 12.28%

    Standard Class:
    - Orders: 15,405
    - Profit Margin: 11.75%
    - Avg Shipping Cost: $40.14
    - Shipping as % of Sales: 8.16%
    ---
    **✅ RECOMMENDATION:**
    - Negotiate better rates with shipping carriers for high-volume modes
    - Implement minimum order values for expensive shipping modes
    - Offer free shipping thresholds to increase average order value
    - Optimize warehouse locations to reduce shipping distances
    """)

with st.expander("6️⃣ SEASONAL & TEMPORAL PATTERNS"):
    st.markdown("""
    **📅 Quarterly Performance:**
    - Q1: $1,991,957 sales, $238,246 profit
    - Q2: $2,873,552 sales, $325,397 profit
    - Q3: $3,478,375 sales, $400,825 profit
    - Q4: $4,298,618 sales, $502,989 profit
    ---
    - 📈 Peak Month: December
    - 📉 Lowest Month: February
    
    ---

    **✅ RECOMMENDATION:**
    - Plan inventory and staffing for peak periods (December)
    - Launch promotions during slow months (February)
    - Implement demand forecasting for better inventory management
    - Consider seasonal product lines and marketing campaigns
    """)

with st.expander("7️⃣ HIGH-VALUE CUSTOMER RETENTION"):
    st.markdown("""
    - 💎 Top 20 customers generate 8.9% of total profit
    - 💰 Combined profit from top 20: $130,880.37

    Top 5 Most Valuable Customers:
    1. Tamara Chand: $8,672.90 profit, 36 orders
    2. Raymond Buch: $8,453.05 profit, 29 orders
    3. Sanjit Chand: $8,205.38 profit, 35 orders
    4. Hunter Lopez: $7,816.57 profit, 24 orders
    5. Bill Eplett: $7,410.01 profit, 42 orders
    ---
    **✅ RECOMMENDATION:**
    - Implement VIP customer program for top 20 customers
    - Assign dedicated account managers to high-value clients
    - Offer exclusive discounts and early access to new products
    - Conduct regular check-ins and satisfaction surveys
    - Protect these relationships - losing one could significantly impact profit
    """)
with st.expander("8️⃣ DISCOUNT IMPACT ANALYSIS"):
    st.markdown("""
    **💸 Discount Impact on Profitability:**
    - 0-10%: 17.23% margin, 3,095 orders
    - 10-20%: 9.86% margin, 4,376 orders
    - 20-30%: -5.53% margin, 843 orders
    - 30%+: -51.27% margin, 5,900 orders
    ---
    **⚠️ Products with >20% discount AND negative profit:**
    1. Cubify CubeX 3D Printer Double Head Print... - $9,239.97 loss
    2. GBC DocuBind P400 Electric Binding System... - $6,859.39 loss
    3. Hoover Stove, White... - $6,714.59 loss
    4. Apple Smart Phone, Full Size... - $6,357.31 loss
    5. Motorola Smart Phone, Cordless... - $5,804.68 loss
    6. Samsung Smart Phone, Cordless... - $5,706.27 loss
    7. Cisco Smart Phone, Cordless... - $5,516.26 loss
    8. Lexmark MX611dhe Monochrome Laser Printer... - $5,269.97 loss
    9. GBC Ibimaster 500 Manual ProClick Binding System... - $5,098.57 loss
    10. Nokia Smart Phone, Full Size... - $5,056.81 loss
    ---
    **✅ RECOMMENDATION:**
    - Review discount policies - heavy discounts may be eroding margins
    - Implement maximum discount thresholds by category
    - Use targeted discounts only for slow-moving inventory
    - Train sales team on value-based selling vs. discount-driven sales
    - A/B test promotions to find optimal discount levels
    """)

with st.expander("📊 EXECUTIVE SUMMARY - TOP 5 ACTION ITEMS"):
    st.markdown("""
    1. 🚨 ADDRESS UNPROFITABLE PRODUCTS
    → 2672 products losing $920,646
    → Action: Review pricing, discontinue chronic loss-makers

    2. 🌍 REPLICATE BEST REGIONAL PRACTICES
    → Canada has 64.3% better margins than worst region
    → Action: Study and implement winning strategies across regions

    3. 💎 PROTECT HIGH-VALUE CUSTOMERS
    → Top 20 customers = 8.9% of profit
    → Action: Launch VIP program, assign account managers

    4. 📦 OPTIMIZE PRODUCT MIX
    → 422 high-volume products have <10% margins
     → Action: Renegotiate supplier prices, adjust pricing strategy

    5. 🚚 IMPROVE SHIPPING EFFICIENCY
    → Average shipping cost: $66.93 per order
    → Action: Negotiate carrier rates, optimize warehouse locations
    """)
with st.expander("💰 POTENTIAL FINANCIAL IMPACT (Conservative Estimates)"):
    st.markdown("""
    1. Eliminate 50% of product losses: +$460,323.08
    2. Improve overall margin by 2%: +$252,850.04
    3. Reduce shipping costs by 10%: +$135,808.57
    ---
    **TOTAL POTENTIAL PROFIT INCREASE**: $848,981.69
    **Current Profit**: $1,467,457.29
    **Projected Profit**: $2,316,438.98
    **Percentage Increase**: 57.9%
    """)
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
