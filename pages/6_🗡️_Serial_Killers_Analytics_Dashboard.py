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
    return df

df = load_data()

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


def clean_data(df):
    df_clean = df.copy()   # ← REQUIRED FIRST LINE

    # Parse dates
    df_clean['Date Apprehended'] = pd.to_datetime(df_clean['Date Apprehended'], errors='coerce')
    df_clean['Born Date'] = pd.to_datetime(df_clean['Born Date'], errors='coerce')
    df_clean['Died Date'] = pd.to_datetime(df_clean['Died Date'], errors='coerce')

    # Derived fields
    df_clean['Active_Years'] = df_clean['End year'] - df_clean['Start year']
    df_clean['Active_Years'] = df_clean['Active_Years'].apply(lambda x: max(x, 0) if pd.notna(x) else np.nan)
    df_clean['Decade'] = (df_clean['Start year'] // 10 * 10).astype('Int64')

    # Victim category
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

# Apply cleaning
df_clean = clean_data(df)


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
min_year = int(df['Start year'].min()) if df_clean['Start year'].notna().any() else 1900
max_year = int(df['Start year'].max()) if df_clean['Start year'].notna().any() else 2020

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)
# Country filter
countries = ['All'] + sorted(df_clean['Country'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", countries)

# Victim category filter
victim_cats = ['All'] + sorted(df_clean['Victim_Category'].unique().tolist())
selected_victim_cat = st.sidebar.selectbox("Select Victim Category", victim_cats)

# Region filter
regions = ['All'] + sorted(df_clean['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)

# Apply filters
filtered_df = df_clean.copy()
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

# Top countries bar chart
country_counts = filtered_df['Country'].value_counts().head(15)
fig1 = px.bar(
    x=country_counts.values,
    y=country_counts.index,
    orientation='h',
    title='Top 15 Countries by Serial Killer Count',
    labels={'x': 'Number of Cases', 'y': 'Country'},
    color=country_counts.values,
    color_continuous_scale='Reds'
)
fig1.update_layout(height=500, showlegend=False)
st.plotly_chart(fig1, width="stretch")

st.markdown("   ")

# Regional distribution
region_counts = filtered_df['Region'].value_counts()
fig2 = px.pie(
    values=region_counts.values,
    names=region_counts.index,
    title='Distribution by Region',
    color_discrete_sequence=px.colors.sequential.Reds_r,
    hole=0.4
)
fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.update_layout(height=500)
st.plotly_chart(fig2, width="stretch")

st.markdown("   ")
# World map
country_victims = df_clean.groupby('Country').agg({
    'Proven victims': 'sum',
    'Name': 'count'
    }).reset_index()
country_victims.columns = ['Country', 'Total_Victims', 'Serial_Killer_Count']
country_victims['Avg_Victims_Per_Killer'] = (
    country_victims['Total_Victims'] / country_victims['Serial_Killer_Count']
    ).round(2)

red_orange_scale = [
    [0.0, "#2b0000"],
    [0.3, "#660000"],
    [0.6, "#cc3300"],
    [0.8, "#ff6600"],
    [1.0, "#ffcc66"]
]
fig3 = px.choropleth(
    country_victims,
    locations='Country',
    locationmode='country names',
    color='Serial_Killer_Count',
    hover_name='Country',
    hover_data={'Total_Victims': True, 'Avg_Victims_Per_Killer': ':.2f'},
    title='Global Distribution Heat Map',
    color_continuous_scale=red_orange_scale,
    labels={'Serial_Killer_Count': 'Number of Serial Killers'},
    template="plotly_dark"
)
fig3.update_layout(height=600)
fig3.update_coloraxes(cmin=0, cmax=country_victims['Serial_Killer_Count'].max())
st.plotly_chart(fig3, uwidth="stretch")

st.markdown("   ")
st.subheader("📅 :green[Temporal Trends]", divider="green")

# Decade analysis
decade_analysis = filtered_df.groupby('Decade').agg({
    'Name': 'count',
    'Proven victims': 'sum'
    }).reset_index()
decade_analysis.columns = ['Decade', 'Serial_Killer_Count', 'Total_Victims']

fig4 = go.Figure()

fig4.add_trace(go.Bar(
    x=decade_analysis['Decade'],
    y=decade_analysis['Serial_Killer_Count'],
    name='Number of Cases',
    marker_color='#2BB570',
    yaxis='y'
))

fig4.add_trace(go.Scatter(
    x=decade_analysis['Decade'],
    y=decade_analysis['Total_Victims'],
    name='Total Victims',
    mode='lines+markers',
    marker_color='#ADFF2F',
    line=dict(width=3),
    yaxis='y2'
))

fig4.update_layout(
    title='Serial Killer Activity by Decade',
    xaxis=dict(title='Decade'),
    yaxis=dict(title='Number of Cases', side='left'),
    yaxis2=dict(title='Total Victims', overlaying='y', side='right'),
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig4, width="stretch")

st.markdown("   ")

# Yearly trend
yearly_stats = filtered_df.groupby('Start year').agg({
    'Name': 'count',
    'Proven victims': 'mean'
}).reset_index()
yearly_stats.columns = ['Year', 'Cases_Started', 'Avg_Victims']

fig5 = px.line(
    yearly_stats,
    x='Year',
    y='Cases_Started',
    title='New Cases Started per Year',
    labels={'Cases_Started': 'Number of Cases', 'Year': 'Year'},
    markers=True
)
fig5.update_traces(line_color='#2BB570')
fig5.update_layout(height=400)
st.plotly_chart(fig5, width="stretch")

st.markdown("   ")

# Active period distribution
active_filtered = filtered_df[filtered_df['Active_Years'].notna() & (filtered_df['Active_Years'] <= 50)]
fig6 = px.histogram(
    active_filtered,
    x='Active_Years',
    nbins=30,
    title='Distribution of Active Period Duration',
    labels={'Active_Years': 'Years Active', 'count': 'Frequency'},
    color_discrete_sequence=['#2BB570']
)
fig6.update_layout(height=400)
st.plotly_chart(fig6, width="stretch")


st.markdown("   ")
st.subheader("💀 :yellow[Victim Analysis]", divider="yellow")

# Victim category distribution
victim_category_counts = filtered_df['Victim_Category'].value_counts()
fig7 = px.pie(
    values=victim_category_counts.values,
    names=victim_category_counts.index,
    title='Distribution by Victim Count Category',
    color_discrete_sequence=px.colors.sequential.Oryel,
    hole=0.4
)
fig7.update_traces(textposition='inside', textinfo='percent+label')
fig7.update_layout(height=500)
st.plotly_chart(fig7, width="stretch")

st.markdown("   ")

# Proven vs Possible victims
scatter_data = filtered_df[
(filtered_df['Proven victims'] < 100) &
(filtered_df['Possible victims'] < 100)
]
fig8 = px.scatter(
scatter_data,
x='Proven victims',
y='Possible victims',
hover_data=['Name', 'Country'],
title='Proven vs. Possible Victims',
labels={'Proven victims': 'Proven Victims', 'Possible victims': 'Possible Victims'},
opacity=0.6,
color='Active_Years',
color_continuous_scale='Viridis'
)
fig8.add_trace(go.Scatter(
x=[0, 100],
y=[0, 100],
mode='lines',
line=dict(dash='dash', color='gray'),
name='Equal Line',
showlegend=True
))
fig8.update_layout(height=500)
st.plotly_chart(fig8, width="stretch")
st.markdown("   ")

# Top deadliest serial killers
st.subheader(":yellow[Top 20 Deadliest Cases]")
top_killers = filtered_df.nlargest(20, 'Proven victims')[
['Name', 'Country', 'Proven victims', 'Start year', 'End year', 'Active_Years']
]

fig9 = px.bar(
top_killers,
x='Proven victims',
y='Name',
orientation='h',
title='Top 20 Deadliest Serial Killers',
labels={'Proven victims': 'Number of Proven Victims', 'Name': 'Serial Killer'},
hover_data=['Country', 'Start year', 'End year'],
color='Proven victims',
color_continuous_scale='YlOrBr'
)
fig9.update_layout(height=700, showlegend=False, yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig9, width="stretch")


st.markdown("   ")
# Victim statistics by region
st.subheader(":yellow[Victim Statistics by Region]")

region_victim_stats = filtered_df.groupby('Region').agg({
    'Proven victims': ['mean', 'median', 'sum', 'count']
}).round(2)

region_victim_stats.columns = [
    'Avg Victims', 'Median Victims', 'Total Victims', 'Case Count'
]
region_victim_stats = region_victim_stats.sort_values('Total Victims', ascending=False)

styled_df = region_victim_stats.style.set_table_styles([
    {'selector': 'th.col_heading',
    'props':[
        ('background-color', '#FFED4B'),
        ('color', '#8B4421'),
        ('font-weight', 'bold'),
        ('border', '1px solid #ffffff'),
        ('padding', '8px')
    ]}
])

st.dataframe(styled_df, width="stretch")

st.markdown("   ")
st.subheader("⚖️ :orange[Criminal Justice]", divider="orange")

# Penalty distribution
penalty_counts = filtered_df['Penalty_Category'].value_counts()
fig10 = px.bar(
x=penalty_counts.index,
y=penalty_counts.values,
title='Distribution of Criminal Penalties',
labels={'x': 'Penalty Type', 'y': 'Number of Cases'},
color=penalty_counts.values,
color_continuous_scale='Fall'
)
fig10.update_layout(height=500, showlegend=False)
st.plotly_chart(fig10, width="stretch")
st.markdown("   ")

# Penalty by decade heatmap
penalty_decade = pd.crosstab(filtered_df['Decade'], filtered_df['Penalty_Category'])
fig11 = px.imshow(
penalty_decade.T,
labels=dict(x="Decade", y="Penalty Type", color="Count"),
title="Penalty Types by Decade",
color_continuous_scale='Peach',
aspect="auto"
)
fig11.update_layout(height=500)
st.plotly_chart(fig11, width="stretch")

# Time to apprehension analysis
df_apprehended = filtered_df[filtered_df['Date Apprehended'].notna()].copy()

if len(df_apprehended) > 0:
    df_apprehended['Apprehension_Year'] = df_apprehended['Date Apprehended'].dt.year
    df_apprehended['Years_To_Catch'] = df_apprehended['Apprehension_Year'] - df_apprehended['Start year']
    df_apprehended = df_apprehended[df_apprehended['Years_To_Catch'] >= 0]

st.subheader(":orange[Time to Apprehension Analysis]")

# Box plot by victim category
fig12 = px.box(
df_apprehended[df_apprehended['Years_To_Catch'] <= 50],
x='Victim_Category',
y='Years_To_Catch',
title='Time to Apprehension by Victim Category',
labels={'Years_To_Catch': 'Years to Catch', 'Victim_Category': 'Victim Category'},
color='Victim_Category',
color_discrete_sequence=px.colors.sequential.Redor
)
fig12.update_layout(height=700, showlegend=False)
fig12.update_xaxes(tickangle=45)
st.plotly_chart(fig12, width="stretch")
st.markdown("   ")

# Distribution of time to catch
fig13 = px.histogram(
df_apprehended[df_apprehended['Years_To_Catch'] <= 50],
x='Years_To_Catch',
nbins=25,
title='Distribution of Years to Apprehension',
labels={'Years_To_Catch': 'Years to Catch', 'count': 'Frequency'},
color_discrete_sequence=px.colors.sequential.Peach
)
fig13.update_layout(height=400)
st.plotly_chart(fig13, width="stretch")
st.markdown("   ")
# Statistics
st.subheader(":orange[Apprehension Statistics]")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        Components.metric_card(
            title="Cases with Apprehension Data",
            value=f"{len(df_apprehended):,}",
            delta="🔚",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Avg Years to Catch",
            value=f"{df_apprehended['Years_To_Catch'].mean():.1f}",
            delta="🔛",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Median Years to Catch",
            value=f"{df_apprehended['Years_To_Catch'].median():.1f}",
            delta="🔜",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Max Years to Catch",
            value=f"{df_apprehended['Years_To_Catch'].max():.0f}",
            delta="🔝",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("   ")

# Penalty severity vs victim count
st.subheader(":orange[Penalty Severity vs Victim Count]")
penalty_victim_analysis = filtered_df.groupby('Penalty_Category')['Proven victims'].agg(['mean', 'median', 'count']).round(2)
penalty_victim_analysis.columns = ['Avg Proven Victims', 'Median Proven Victims', 'Case Count']
penalty_victim_analysis = penalty_victim_analysis.sort_values('Avg Proven Victims', ascending=False)
st.dataframe(penalty_victim_analysis, width="stretch")

st.markdown("   ")
st.subheader("📋 :red[Data Explorer]", divider="red")

# Search functionality
st.subheader("🔍 :rainbow[Search Serial Killers]")
search_term = st.text_input("Search by name, nickname, or location:", "")

if search_term:
    search_results = filtered_df[
        filtered_df['Name'].str.contains(search_term, case=False, na=False) |
        filtered_df['Nicknames'].str.contains(search_term, case=False, na=False) |
        filtered_df['Born Location'].str.contains(search_term, case=False, na=False) |
        filtered_df['Country'].str.contains(search_term, case=False, na=False)
    ]
    st.write(f"Found {len(search_results)} result(s)")

    if len(search_results) > 0:
        display_cols = [
            'Name', 'Nicknames', 'Country', 'Start year', 'End year',
            'Proven victims', 'Possible victims', 'Criminal Penalty'
        ]
        st.dataframe(search_results[display_cols], width="stretch")

# Full dataset view
st.subheader("📊 :rainbow[Full Dataset]")

# Column selector
all_columns = filtered_df.columns.tolist()
default_columns = ['Name', 'Country', 'Start year', 'End year', 'Proven victims',
'Possible victims', 'Criminal Penalty', 'Victim_Category']
selected_columns = st.multiselect(
    "Select columns to display:",
    all_columns,
    default=default_columns
)
if selected_columns:
    # Sorting options
    col1, col2 = st.columns(2)
    with col1:
        sort_column = st.selectbox("Sort by:", selected_columns)
    with col2:
        sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
        # Display sorted data
    ascending = (sort_order == "Ascending")
    sorted_df = filtered_df[selected_columns].sort_values(by=sort_column, ascending=ascending)

st.dataframe(sorted_df, width="stretch", height=600)

# Download button
csv = sorted_df.to_csv(index=False).encode('utf-8')
st.download_button(
label="📥 Download Filtered Data as CSV",
data=csv,
file_name=f'serial_killers_filtered_{datetime.now().strftime("%Y%m%d")}.csv',
mime='text/csv'
)

# Summary statistics
st.subheader("📈 :blue[Summary Statistics]", divider='blue')

col1, col2 = st.columns(2)
with col1:
    st.write("Numerical Summary")
    numerical_summary = filtered_df[['Start year', 'End year', 'Proven victims',
    'Possible victims', 'Active_Years']].describe()
st.dataframe(numerical_summary, width="stretch")

with col2:
    st.write("Categorical Summary")
    categorical_stats = pd.DataFrame({
        'Unique Countries': [filtered_df['Country'].nunique()],
        'Unique Regions': [filtered_df['Region'].nunique()],
        'Cases with Death Penalty': [(filtered_df['Penalty_Category'] == 'Death Penalty').sum()],
        'Cases with Life Sentence': [(filtered_df['Penalty_Category'] == 'Life Imprisonment').sum()],
        'Cases with Apprehension Date': [filtered_df['Date Apprehended'].notna().sum()],
        'Complete Records': [filtered_df[['Name', 'Country', 'Start year', 'Proven victims']].notna().all(axis=1).sum()]
}).T
categorical_stats.columns = ['Count']
st.dataframe(categorical_stats, width="stretch")

# Footer
st.markdown("---")
st.markdown("""

Serial Killers Analytics Dashboard


Data Source: serial_killers.csv | Total Records: {}


Dashboard built with Streamlit & Plotly



""".format(len(df)), unsafe_allow_html=True)

# Sidebar - Additional Info
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dashboard Info")
st.sidebar.info(f"""
Filtered Results:
- Cases: {len(filtered_df):,}
- Countries: {filtered_df['Country'].nunique()}
- Total Victims: {int(filtered_df['Proven victims'].sum()):,}
- Year Range: {int(filtered_df['Start year'].min()) if len(filtered_df) > 0 else 'N/A'} - {int(filtered_df['Start year'].max()) if len(filtered_df) > 0 else 'N/A'}
""")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.write("""
This dashboard provides comprehensive analytics on serial killer cases worldwide, including:
- Geographic distribution patterns
- Temporal trends and historical analysis
- Victim statistics and patterns
- Criminal justice outcomes
- Interactive data exploration tools
""")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Use filters above to narrow down your analysis")