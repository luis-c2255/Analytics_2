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
    value=(df['date'].min(), df['date'].max()),
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
st.markdown("---")
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
st.markdown("---")
# Monthly breakdown
col1, col2 = st.columns(2)

with col1:
    monthly_avg = filtered_df.groupby('month_name').agg({
            'average temperature': 'mean',
            'maximum temperature': 'max',
            'minimum temperature': 'min'
        }).reindex([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
                x=monthly_avg.index,
                y=monthly_avg['average temperature'],
                name='Avg Temperature',
                marker_color='lightseagreen',
                hovertemplate='%{x}Avg: %{y:.1f}°F'
            ))
    fig_monthly.update_layout(
                title="Average Temperature by Month",
                xaxis_title="Month",
                yaxis_title="Temperature (°F)",
                height=400,
            )
    st.plotly_chart(fig_monthly, width="stretch")

# Temperature distribution
with col2:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=filtered_df['average temperature'],
        nbinsx=30,
        marker_color='coral',
        name='Distribution',
        hovertemplate='Temperature: %{x}°FCount: %{y}'
    ))
    fig_dist.update_layout(
        title="Temperature Distribution",
        xaxis_title="Temperature (°F)",
        yaxis_title="Frequency",
        height=400,
    )
    st.plotly_chart(fig_dist, width="stretch")
st.markdown("---")
st.markdown(
    Components.page_header(
        "🌧️ Precipitation Analysis"
    ), unsafe_allow_html=True
)
st.markdown("---")
with st.container():
    fig_precip = go.Figure()
    fig_precip.add_trace(go.Bar(
        x=filtered_df['date'],
        y=filtered_df['precipitation'],
        marker_color='skyblue',
        name='Precipitation',
        hovertemplate='Date: %{x}Precipitation: %{y:.2f}"'
    ))
    fig_precip.update_layout(
        title="Daily Precipitation",
        xaxis_title="Date",
        yaxis_title="Precipitation (inches)",
        height=400,
    )
    st.plotly_chart(fig_precip, width="stretch")
st.markdown("---")
st.subheader("📈 :red[Precipitation Stats]", divider="red")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    rainy_df = filtered_df[filtered_df['precipitation'] > 0]
    st.markdown(
        Components.metric_card(
            title="Total Precipitation",
            value=f"{filtered_df['precipitation'].sum():.2f}",
            delta="🌦️",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    rainy_df = filtered_df[filtered_df['precipitation'] > 0]
    st.markdown(
        Components.metric_card(
            title="Rainy Days",
            value=f"{len(rainy_df)} days",
            delta="☔",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    rainy_df = filtered_df[filtered_df['precipitation'] > 0]
    st.markdown(
        Components.metric_card(
            title="Avg per Rainy Day",
            value=f"{rainy_df['precipitation'].mean():.2f}" if len(rainy_df) > 0 else "N/A",
            delta="💧",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    rainy_df = filtered_df[filtered_df['precipitation'] > 0]
    st.markdown(
        Components.metric_card(
            title="Wettest Day",
            value=f"{filtered_df['precipitation'].max():.2f}",
            delta="⛈️",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col5:
    rainy_df = filtered_df[filtered_df['precipitation'] > 0]
    st.markdown(
        Components.metric_card(
            title="Dry Days",
            value=f"{(filtered_df['precipitation'] == 0).sum()} days",
            delta="🌤️",
            card_type="info"
        ), unsafe_allow_html=True
    )

st.markdown("---")
# Monthly precipitation totals
monthly_precip = filtered_df.groupby('month_name')['precipitation'].sum().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])
fig_monthly_precip = px.bar(
    x=monthly_precip.index,
    y=monthly_precip.values,
    labels={'x': 'Month', 'y': 'Total Precipitation (inches)'},
    title='Monthly Precipitation Totals',
    color=monthly_precip.values,
    color_continuous_scale='Blues'
)
fig_monthly_precip.update_layout(height=400)
st.plotly_chart(fig_monthly_precip, width="stretch")
st.markdown("---")
st.markdown(
    Components.page_header(
        "❄️ Snow Patterns"
    ), unsafe_allow_html=True
)
st.subheader(":orange[Snow Analysis]", divider="orange")
# Snow depth over time
fig_snow = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Snow Depth Over Time', 'Daily Snowfall'),
    vertical_spacing=0.15
)
fig_snow.add_trace(
    go.Scatter(
        x=filtered_df['date'],
        y=filtered_df['snow depth'],
        fill='tozeroy',
        name='Snow Depth',
        line=dict(color='lightblue', width=2),
        hovertemplate='Date: %{x}Depth: %{y:.1f}"'
    ),
    row=1, col=1
)
fig_snow.add_trace(
    go.Bar(
        x=filtered_df['date'],
        y=filtered_df['snow fall'],
        name='Snowfall',
        marker_color='steelblue',
        hovertemplate='Date: %{x}Snowfall: %{y:.1f}"'
    ),
    row=2, col=1
)
fig_snow.update_xaxes(title_text="Date", row=2, col=1)
fig_snow.update_yaxes(title_text="Snow Depth (inches)", row=1, col=1)
fig_snow.update_yaxes(title_text="Snowfall (inches)", row=2, col=1)

fig_snow.update_layout(
    height=600,
    showlegend=True,
    hovermode='x unified'
)
st.plotly_chart(fig_snow, width="stretch")
st.markdown("---")
# Snow statistics
col1, col2, col3 = st.columns(3)

snowy_days = filtered_df[filtered_df['snow fall'] > 0]
with col1:
    st.markdown("#### ❄️ Snowfall Stats")
    st.write(f"Total Snowfall: {filtered_df['snow fall'].sum():.2f}\"")
    st.write(f"Snowy Days: {len(snowy_days)}")
    if len(snowy_days) > 0:
        st.write(f"Avg per Snowy Day: {snowy_days['snow fall'].mean():.2f}\"")

with col2:
    st.markdown("#### 🏔️ Snow Depth Stats")
    st.write(f"Max Depth: {filtered_df['snow depth'].max():.2f}\"")
    st.write(f"Avg Depth: {filtered_df['snow depth'].mean():.2f}\"")
    days_with_snow = (filtered_df['snow depth'] > 0).sum()
    st.write(f"Days with Snow on Ground: {days_with_snow}")

with col3:
    st.markdown("#### 🌨️ Extreme Events")
    if len(snowy_days) > 0:
        snowiest = snowy_days.loc[snowy_days['snow fall'].idxmax()]
        st.write(f"Snowiest Day:")
        st.write(f"{snowiest['date'].strfdate('%B %d, %Y')}")
        st.write(f"{snowiest['snow fall']:.2f}\" of snow")
        deepest = filtered_df.loc[filtered_df['snow depth'].idxmax()]
        st.write(f"Deepest Snow:")
        st.write(f"{deepest['date'].strftime('%B %d, %Y')}")
        st.write(f"{deepest['snow depth']:.2f}\" depth")
st.markdown("---")
st.markdown(
    Components.page_header(
        "📊 Statistical Insights"
    ), unsafe_allow_html=True
)
st.markdown("---")
st.subheader(":rainbow[Advanced Statistical Analysis]", divider="rainbow")
col1, col2 = st.columns(2)
with col1:
    # Correlation heatmap
    st.markdown("#### 🔗 Variable Correlations")
    numeric_cols = ['maximum temperature', 'minimum temperature',
    'average temperature', 'precipitation', 'snow fall', 'snow depth']
    corr_matrix = filtered_df[numeric_cols].corr()
    fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}Correlation: %{z:.3f}'
        ))
        fig_corr.update_layout(
            height=450,
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, width="stretch")

with col2:
    # Temperature variability
    st.markdown("#### 📊 Temperature Variability")
    fig_var = go.Figure()
    
    fig_var.add_trace(go.Box(
        y=filtered_df['temp_range'],
        name='Daily Range',
        marker_color='lightcoral',
        boxmean='sd'
        ))
        fig_var.update_layout(
            title="Daily Temperature Range Distribution",
            yaxis_title="Temperature Range (°F)",
            height=450,
        )
        st.plotly_chart(fig_var, width="stretch")
st.markdown("---")
st.write(f"Average Daily Range: {filtered_df['temp_range'].mean():.1f}°F")
st.write(f"Most Variable Day: {filtered_df['temp_range'].max():.0f}°F")
st.write(f"Least Variable Day: {filtered_df['temp_range'].min():.0f}°F")
# Seasonal comparison
st.markdown("#### 🍂 Seasonal Breakdown")

seasons = {
'Winter': [12, 1, 2],
'Spring': [3, 4, 5],
'Summer': [6, 7, 8],
'Fall': [9, 10, 11]
}

seasonal_data = []
for season, months in seasons.items():
season_df = filtered_df[filtered_df['month'].isin(months)]
if len(season_df) > 0:
seasonal_data.append({
'Season': season,
'Avg Temperature (°F)': round(season_df['average temperature'].mean(), 1),
'Total Precipitation (")': round(season_df['precipitation'].sum(), 2),
'Total Snowfall (")': round(season_df['snow fall'].sum(), 2),
'Rainy Days': season_df['is_rainy'].sum(),
'Snowy Days': season_df['is_snowy'].sum()
})

seasonal_df = pd.DataFrame(seasonal_data)

fig_seasonal = go.Figure()

fig_seasonal.add_trace(go.Bar(
name='Avg Temperature',
x=seasonal_df['Season'],
y=seasonal_df['Avg Temperature (°F)'],
marker_color='orange',
yaxis='y',
hovertemplate='%{x}Temp: %{y:.1f}°F'
))

fig_seasonal.add_trace(go.Scatter(
name='Total Precipitation',
x=seasonal_df['Season'],
y=seasonal_df['Total Precipitation (")'],
marker_color='blue',
yaxis='y2',
mode='lines+markers',
line=dict(width=3),
hovertemplate='%{x}Precip: %{y:.2f}"'
))

fig_seasonal.update_layout(
title='Seasonal Temperature vs Precipitation',
yaxis=dict(title='Temperature (°F)', side='left'),
yaxis2=dict(title='Precipitation (inches)', side='right', overlaying='y'),
height=400,
template='plotly_white',
hovermode='x unified'
)

st.plotly_chart(fig_seasonal, use_container_width=True)

# Display seasonal table
st.dataframe(seasonal_df, use_container_width=True, hide_index=True)

st.markdown(
    Components.page_header(
        "🔍 Custom Explorer"
    ), unsafe_allow_html=True
)
# Variable selection
col1, col2 = st.columns(2)

with col1:
x_var = st.selectbox(
"Select X-axis variable",
options=['date', 'average temperature', 'precipitation', 'snow fall', 'snow depth', 'temp_range'],
index=0
)

with col2:
y_var = st.selectbox(
"Select Y-axis variable",
options=['average temperature', 'maximum temperature', 'minimum temperature',
'precipitation', 'snow fall', 'snow depth', 'temp_range'],
index=0
)

# Chart type
chart_type = st.radio(
"Chart Type",
options=['Scatter', 'Line', 'Bar'],
horizontal=True
)

# Color by option
color_by = st.selectbox(
"Color by",
options=[None, 'month_name', 'day_of_week', 'is_rainy', 'is_snowy'],
index=0
)

# Generate custom chart
if chart_type == 'Scatter':
fig_custom = px.scatter(
filtered_df,
x=x_var,
y=y_var,
color=color_by,
title=f'{y_var} vs {x_var}',
hover_data=['date', 'average temperature', 'precipitation']
)
elif chart_type == 'Line':
fig_custom = px.line(
filtered_df,
x=x_var,
y=y_var,
color=color_by,
title=f'{y_var} over {x_var}'
)
else: # Bar
fig_custom = px.bar(
filtered_df,
x=x_var,
y=y_var,
color=color_by,
title=f'{y_var} by {x_var}'
)

fig_custom.update_layout(height=500, template='plotly_white')
st.plotly_chart(fig_custom, use_container_width=True)

# Data table explorer
st.markdown("####📋 Raw Data Explorer")

# Column selector
display_cols = st.multiselect(
"Select columns to display",
options=filtered_df.columns.tolist(),
default=['date', 'maximum temperature', 'minimum temperature',
'average temperature', 'precipitation', 'snow fall']
)

# Show extreme values option
show_extremes = st.checkbox("Show only extreme weather days", value=False)

if show_extremes:
extreme_df = filtered_df[
(filtered_df['maximum temperature'] > filtered_df['maximum temperature'].quantile(0.9)) |
(filtered_df['minimum temperature'] < filtered_df['minimum temperature'].quantile(0.1)) |
(filtered_df['precipitation'] > filtered_df['precipitation'].quantile(0.9)) |
(filtered_df['snow fall'] > 0)
]
display_df = extreme_df[display_cols] if display_cols else extreme_df
st.write(f"Showing {len(display_df)} extreme weather days")
else:
display_df = filtered_df[display_cols] if display_cols else filtered_df
st.write(f"Showing {len(display_df)} days")

# Display dataframe with sorting
st.dataframe(
display_df.style.background_gradient(cmap='coolwarm', subset=[col for col in display_cols if 'temperature' in col.lower()]),
use_container_width=True,
height=400
)

# Download button
csv = display_df.to_csv(index=False).encode('utf-8')
st.download_button(
label="📥 Download Filtered Data as CSV",
data=csv,
file_name=f'nyc_weather_filtered_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
mime='text/csv'
)

# Footer with insights summary
st.markdown("---")
st.markdown("### 🎯 Key Takeaways from Analysis")

col1, col2, col3 = st.columns(3)

with col1:
st.markdown("#### Temperature Patterns")
warmest_month = filtered_df.groupby('month_name')['average temperature'].mean().idxmax()
coldest_month = filtered_df.groupby('month_name')['average temperature'].mean().idxmin()
st.write(f"• Warmest month: {warmest_month}")
st.write(f"• Coldest month: {coldest_month}")
st.write(f"• Avg temperature variability: {filtered_df['temp_range'].mean():.1f}°F")

with col2:
st.markdown("#### Precipitation Insights")
wettest_month = filtered_df.groupby('month_name')['precipitation'].sum().idxmax()
rainy_pct = (filtered_df['is_rainy'].sum() / len(filtered_df)) * 100
st.write(f"• Wettest month: {wettest_month}")
st.write(f"• Rainy days: {rainy_pct:.1f}% of year")
st.write(f"• Total precipitation: {filtered_df['precipitation'].sum():.2f}\"")

with col3:
st.markdown("#### Snow Summary")
if filtered_df['snow fall'].sum() > 0:
snowiest_month = filtered_df.groupby('month_name')['snow fall'].sum().idxmax()
st.write(f"• Snowiest month: {snowiest_month}")
st.write(f"• Total snowfall: {filtered_df['snow fall'].sum():.2f}\"")
st.write(f"• Snowy days: {filtered_df['is_snowy'].sum()} days")
else:
	st.write("• No snow in selected period")

# Sidebar - Additional Information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About This Dashboard")
st.sidebar.info("""
This interactive dashboard analyzes NYC weather data from 2016.

Features:
- Real-time filtering by date and month
- Interactive visualizations
- Statistical analysis
- Correlation insights
- Custom data exploration
- Export capabilities

Data Source: NYC Weather 2016
Total Records: 366 days
""")

st.sidebar.markdown("### 💡 Usage Tips")
st.sidebar.markdown("""
1. Use filters to focus on specific periods
2. Hover over charts for detailed info
3. Switch between tabs for different analyses
4. Use Custom Explorer for ad-hoc queries
5. Download filtered data for further analysis
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
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
