import streamlit as st
import pandas as pd
import numpy as np
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

st.sidebar.markdown("---")
st.sidebar.markdown("📊 Dashboard shows comprehensive stock market analysis with technical indicators")

st.markdown(
    Components.page_header(
        "🆙 Yahoo Stock Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("   ")
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

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    subplot_titles=('Stock Price', 'Trading Volume')
)
# Price chart
if chart_type == 'Candlestick':
    fig.add_trace(
        go.Candlestick(
            x=filtered_df['Date'],
            open=filtered_df['Open'],
            high=filtered_df['High'],
            low=filtered_df['Low'],
            close=filtered_df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
else:
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
# Add moving averages
if '50-Day MA' in ma_options:
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Close_MA_50'],
            name='50-Day MA',
            line=dict(color='orange', width=1.5)
        ),
        row=1, col=1
    )
if '200-Day MA' in ma_options:
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Close_MA_200'],
            name='200-Day MA',
            line=dict(color='red', width=1.5)
        ),
        row=1, col=1
    )
    # Volume
    colors = ['red' if row['Close'] < row['Open'] else 'green' for idx, row in filtered_df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=filtered_df['Date'],
            y=filtered_df['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text='Price ($)', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)

    st.plotly_chart(fig, width="stretch")

st.markdown("   ")
st.subheader("📈 :blue[Daily Returns Distribution]", divider="blue")

fig_returns = go.Figure()
fig_returns.add_trace(go.Histogram(
    x=filtered_df['Daily_Return'].dropna(),
    nbinsx=50,
    marker_color='steelblue',
    name='Daily Returns'
))
fig_returns.update_layout(
    xaxis_title='Daily Return (%)',
    yaxis_title='Frequency',
    height=400,
    showlegend=False
)
st.plotly_chart(fig_returns, width="stretch")

st.markdown("   ")
st.subheader("↩️ :blue[Return Statistics]", divider="blue")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    current_price = filtered_df['Close'].iloc[-1]
    st.markdown(
        Components.metric_card(
            title="Mean",
            value=f"{filtered_df['Daily_Return'].mean():.4f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col2:
    total_return = ((filtered_df['Close'].iloc[-1] - filtered_df['Close'].iloc[0]) / filtered_df['Close'].iloc[0]) * 100
    st.markdown(
        Components.metric_card(
            title="Median",
            value=f"{filtered_df['Daily_Return'].median():.4f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    avg_volume = filtered_df['Volume'].mean()
    st.markdown(
        Components.metric_card(
            title="Std Dev",
            value=f"{filtered_df['Daily_Return'].std():.4f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    volatility = filtered_df['Daily_Return'].std()
    st.markdown(
        Components.metric_card(
            title="Best Day",
            value=f"{filtered_df['Daily_Return'].max():.2f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col5:
    volatility = filtered_df['Daily_Return'].std()
    st.markdown(
        Components.metric_card(
            title="Worst Day",
            value=f"{filtered_df['Daily_Return'].min():.2f}%",
            delta="",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader("⚡ :orange[Market Volatility (30-Day Rolling)]", divider="orange")

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=filtered_df['Date'],
    y=filtered_df['Volatility_30D'],
    mode='lines',
    fill='tozeroy',
    name='Volatility',
    line=dict(color='purple', width=2)
))
# Add threshold line
vol_threshold = filtered_df['Volatility_30D'].quantile(0.90)
fig_vol.add_hline(
    y=vol_threshold,
    line_dash='dash',
    line_color='red',
    annotation_text=f"90th Percentile: {vol_threshold:.2f}%"
)
fig_vol.update_layout(
    xaxis_title='Date',
    yaxis_title='Volatility (%)',
    height=400,
    showlegend=False
)
st.plotly_chart(fig_vol, width="stretch")

st.markdown("   ")
st.subheader("💣 :violet[Volatility Analysis]", divider="violet")
# High volatility periods
high_vol_count = len(filtered_df[filtered_df['Volatility_30D'] >vol_threshold])
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.metric_card(
            title="Current 30D Volatility",
            value=f"{filtered_df['Volatility_30D'].iloc[-1]:.4f}%",
            delta="",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Average Volatility",
            value=f"{filtered_df['Volatility_30D'].mean():.4f}%",
            delta="",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="High Volatility Days",
            value=f"{high_vol_count} ({high_vol_count/len(filtered_df)*100:.1f}%)",
            delta="",
            card_type="warning"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader("🎯 :red[Support & Resistance Levels]", divider="red")

# Support/Resistance and Monthly Performance
# Use last 200 days for clearer visualization
recent_data = filtered_df.tail(200)
fig_sr = go.Figure()
fig_sr.add_trace(go.Scatter(
    x=recent_data['Date'],
    y=recent_data['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='blue', width=2)
))
fig_sr.add_trace(go.Scatter(
    x=recent_data['Date'],
    y=recent_data['Support_50D'],
    mode='lines',
    name='Support (50D)',
    line=dict(color='green', width=2, dash='dash')
))
fig_sr.add_trace(go.Scatter(
    x=recent_data['Date'],
    y=recent_data['Resistance_50D'],
    mode='lines',
    name='Resistance (50D)',
    line=dict(color='red', width=2, dash='dash')
))
fig_sr.update_layout(
    xaxis_title='Date',
    yaxis_title='Price ($)',
    height=400,
    hovermode='x unified'
)
st.plotly_chart(fig_sr, width="stretch")

st.markdown("   ")
st.subheader("📑 :green[Current Technical Levels]", divider="green")
# Current levels
recent_support = filtered_df['Support_50D'].iloc[-1]
recent_resistance = filtered_df['Resistance_50D'].iloc[-1]
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.metric_card(
            title="Current Price",
            value=f"${current_price:,.2f}",
            delta="",
            card_type="error"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Support",
            value=f"${recent_support:,.2f} ({((current_price - recent_support)/current_price)*100:.2f}% away)",
            delta="",
            card_type="error"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Resistance",
            value=f"${recent_resistance:,.2f} ({((recent_resistance - current_price)/current_price)*100:.2f}% away)",
            delta="",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader("📅 :blue[Monthly Performance Analysis]", divider="blue")
# Calculate monthly returns
month_names = ['January', 'February', 'March', 'April', 'May', 'June',
'July', 'August', 'September', 'October', 'November', 'December']

monthly_returns = filtered_df.groupby('Month')['Daily_Return'].mean().reindex(range(1, 13))

fig_monthly = go.Figure()
colors_monthly = ['green' if x > 0 else 'red' for x in monthly_returns]

fig_monthly.add_trace(go.Bar(
    x=month_names,
    y=monthly_returns,
    marker_color=colors_monthly,
    text=[f"{x:.3f}%" for x in monthly_returns],
    textposition='outside'
))
fig_monthly.update_layout(
    xaxis_title='Month',
    yaxis_title='Average Daily Return (%)',
    height=400,
    showlegend=False
)
st.plotly_chart(fig_monthly, width="stretch")
st.markdown("   ")
st.subheader(":rainbow[Seasonal Patterns]", divider="rainbow")
col1, col2 = st.columns(2)
# Best and worst months
best_month = month_names[monthly_returns.idxmax() - 1]
worst_month = month_names[monthly_returns.idxmin() - 1]
with col1:
    st.success(f"- Best Month: {best_month} ({monthly_returns.max():.3f}%)")
with col2:
    st.error(f"- Worst Month: {worst_month} ({monthly_returns.min():.3f}%)")

st.markdown("   ")
st.subheader("💹 :green[Volume vs Price Movement]", divider="green")
fig_scatter = go.Figure()
colors_scatter = ["green" if x > 0 else 'red' for x in filtered_df['Daily_Return']]
fig_scatter.add_trace(go.Scatter(
    x=filtered_df['Volume'],
    y=filtered_df['Daily_Return'].abs(),
    mode='markers',
    marker=dict(
        color=colors_scatter,
        size=5,
        opacity=0.6
    ),
    text=filtered_df['Date'].dt.strftime('%Y-%m-%d'),
    hovertemplate='Date: %{text} Volume: %{x:,.0f} Abs Return: %{y:.2f}%'
))
fig_scatter.update_layout(
    xaxis_title='Trading Volume',
    yaxis_title='Absolute Daily Return (%)',
    height=400,
    showlegend=False
)
st.plotly_chart(fig_scatter, width="stretch")

# Correlation Analysis
vol_return_corr = filtered_df[['Volume', 'Daily_Return']].corr().iloc[0, 1]
st.subheader(":rainbow[Volume Insights]", divider="rainbow")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"- Correlation (Volume vs Return): {vol_return_corr:.4f}")
with col2:
    st.warning(f"- High volume often indicates institutional activity")
with col3:
    st.success(f"- Max Volume Day: {filtered_df.loc[filtered_df['Volume'].idxmax(), 'Date'].strftime('%Y-%m-%d')}")

st.markdown("   ")
st.subheader("🔗 :yellow[Feature Correlation Matrix]", divider="yellow")

# Select key features for correlation
corr_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Price_Range']
corr_matrix = filtered_df[corr_features].corr()

fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=corr_matrix.values,
    texttemplate='%{text:.2f}',
    textfont={'size': 10},
    colorbar=dict(title='Correlation')
))
fig_corr.update_layout(
    height=400,
    xaxis={'side': 'bottom'}
)
st.plotly_chart(fig_corr, width="stretch")

st.subheader(":rainbow[Correlation Insights]", divider="rainbow")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("Strong correlations between OHLC prices (expected)")
with col2:
    st.info("Volume often independent of price direction")
with col3:
    st.info("Price_Range indicates daily volatility")

st.markdown("   ")
st.subheader("🔍 :violet[Advanced Analytics]", divider="violet")

tab1, tab2, tab3 = st.tabs(["📊 Year-over-Year", "🎲 Risk Metrics", "📋 Data Table"])
with tab1:
    # Yearly performance
    yearly_data = filtered_df.groupby('Year').agg({
        'Close': ['first', 'last'],
        'Volume': 'mean',
        'Daily_Return': ['mean', 'std']
    }).round(2)
    yearly_data.columns = ['Open Price', 'Close Price', 'Avg Volume', 'Avg Return (%)', 'Volatility (%)']
    yearly_data['YoY Return (%)'] = ((yearly_data['Close Price'] - yearly_data['Open Price']) / yearly_data['Open Price'] * 100).round(2)
    st.write("Yearly Performance Summary:")
    st.dataframe(yearly_data, width="stretch")
    # Yearly returns chart
    fig_yearly = go.Figure()
    colors_yearly = ['green' if x > 0 else 'red' for x in yearly_data['YoY Return (%)']]
    fig_yearly.add_trace(go.Bar(
        x=yearly_data.index,
        y=yearly_data['YoY Return (%)'],
        marker_color=colors_yearly,
        text=[f"{x:.2f}%" for x in yearly_data['YoY Return (%)']],
        textposition='outside'
    ))
    fig_yearly.update_layout(
        title='Year-over-Year-Returns',
        xaxis_title='Year',
        yaxis_title='Annual Return (%)',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig_yearly, width="stretch")
with tab2:
    st.write("Risk & Performance Metrics:")
    # Calculate risk metrics
    returns = filtered_df['Daily_Return'].dropna()
    # Sharpe Ratio (assuming 0% risk-free rate, annualized)
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    # Maximum Drawdown
    cumulative = (1 + returns / 100).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    # Value at Risk (VaR) - 95% confidence
    var_95 = np.percentile(returns, 5)
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Win Rate
    win_rate = (len(returns[returns > 0]) / len(returns)) * 100
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio (Annualized)", f"{sharpe_ratio:.4f}")
        st.metric("Sortino Ratio (Annualized)", f"{sortino_ratio:.4f}")
    with col2:
        st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
        st.metric("Value at Risk (95%)", f"{var_95:.2f}%")
    with col3:
        st.metric("Win Rate", f"{win_rate:.2f}%")
        st.metric("Avg Win/Loss Ratio", f"{abs(returns[returns > 0].mean() / returns[returns < 0].mean()):.2f}")
    # Drawdown chart
    st.write("Drawdown Over Time:")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=drawdown,
        mode='lines',
        fill='tozeroy',
        line=dict(color='red', width=1.5),
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    fig_dd.update_layout(
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig_dd, width="stretch")
    st.info("""
    Metric Explanations:
    - Sharpe Ratio: Risk-adjusted return (higher is better,>1 is good, >2 is excellent)
    - Sortino Ratio: Like Sharpe but only considers downside volatility
    - Maximum Drawdown: Largest peak-to-trough decline (lower is better)
    - VaR 95%: Expected maximum loss on 95% of trading days
    - Win Rate: Percentage of days with positive returns
    """)
with tab3:
    st.write("Raw Data Table (Filtered by Date Range):")
    # Display options
    show_columns = st.multiselect(
        "Select columns to display:",
        options=filtered_df.columns.tolist(),
        default=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']
    )
    # Number of rows to display
    n_rows = st.slider("Number of rows to display:", min_value=10, max_value=len(filtered_df), value=50)
    # Sort order
    sort_order = st.radio("Sort by:", options=['Most Recent First', 'Oldest First', 'Highest Return', 'Lowest Return'])
    display_df = filtered_df[show_columns].copy()
    if sort_order == 'Most Recent First':
        display_df = display_df.sort_values('Date', ascending=False)
    elif sort_order == 'Oldest First':
        display_df = display_df.sort_values('Date', ascending=True)
    elif sort_order == 'Highest Return':
        if 'Daily_Return' in show_columns:
            display_df = display_df.sort_values('Daily_Return', ascending=False)
        elif sort_order == 'Lowest Return':
            if 'Daily_Return' in show_columns:
                display_df = display_df.sort_values('Daily_Return', ascending=True)
                st.dataframe(display_df.head(n_rows), width="stretch")

                # Download button
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv,
                    file_name=f"stock_data_{date_range[0]}_{date_range[1]}.csv",
                    mime="text/csv"
                )
st.markdown("   ")
st.subheader("💡 :yellow[Key Insights Summary]", divider="yellow")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("📈 Price Trends")
    if total_return > 0:
        st.success(f"✅ Positive return of {total_return:.2f}% over selected period")
    else:
        st.error(f"❌ Negative return of {total_return:.2f}% over selected period")
    # Bull vs Bear days
    bull_days = len(filtered_df[filtered_df['Close'] > filtered_df['Close_MA_200']])
    bear_days = len(filtered_df) - bull_days
    st.write(f"- Bull Market: {bull_days} days ({bull_days/len(filtered_df)*100:.1f}%)")
    st.write(f"- Bear Market: {bear_days} days ({bear_days/len(filtered_df)*100:.1f}%)")
with col2:
    st.write("⚡ Volatility Profile")
    current_vol = filtered_df['Volatility_30D'].iloc[-1]
    avg_vol = filtered_df['Volatility_30D'].mean()
    if current_vol > avg_vol * 1.2:
        st.warning(f"⚠️ Current volatility ({current_vol:.2f}%) is elevated")
    elif current_vol < avg_vol * 0.8:
        st.info(f"📉 Current volatility ({current_vol:.2f}%)is below average")
    else:
        st.success(f"✅ Current volatility ({current_vol:.2f}%) is normal")
        st.write(f"- Average Volatility: {avg_vol:.2f}%")
        st.write(f"- Best Day Return: {filtered_df['Daily_Return'].max():.2f}%")
        st.write(f"- Worst Day Return: {filtered_df['Daily_Return'].min():.2f}%")
with col3:
    st.write("🎯 Trading Signals")
    # Simple technical signals
    last_close = filtered_df['Close'].iloc[-1]
    last_ma50 = filtered_df['Close_MA_50'].iloc[-1]
    last_ma200 = filtered_df['Close_MA_200'].iloc[-1]
    
    if last_close > last_ma50 and last_close > last_ma200:
        st.success("🟢 Bullish: Price above both MAs")
    elif last_close < last_ma50 and last_close < last_ma200:
        st.error("🔴 Bearish: Price below both MAs")
    else:
        st.warning("🟡 Mixed: Price between MAs")
        
        # Support/Resistance proximity
        support_dist = ((last_close - recent_support) / last_close) * 100
        resistance_dist = ((recent_resistance - last_close) / last_close) * 100

        if support_dist < 2:
            st.info("📍 Near support level")
        elif resistance_dist < 2:
            st.info("📍 Near resistance level")
            
            st.write(f"- Distance to Support: {support_dist:.2f}%")
            st.write(f"- Distance to Resistance: {resistance_dist:.2f}%")
st.markdown("   ")
st.subheader("📋 :orange[Actionable Business Recommendations]", divider="orange")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        Components.insight_box(
            "For Traders:",
            """
            <ol style="margin: 0; padding-left: 20px;">
                <li>Use the 50-day and 200-day moving averages as trend indicators</li>
                <li>Monitor volatility spikes (30-day rolling std > 90th percentile) for risk management</li>
                <li>Pay attention to support/resistance levels for entry/exit points</li>
                <li>High volume days often precede major price movements</li>
            </ol>
            """,
            "info"
        ),
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
            Components.insight_box(
                "For Portfolio Managers:",
                """
                <ol style="margin: 0; padding-left: 20px;">
                    <li>Sharpe Ratio provides risk-adjusted return comparison</li>
                    <li>Maximum Drawdown helps size positions appropriately</li>
                    <li>Monthly seasonality can inform timing strategies</li>
                    <li>Win rate analysis helps set realistic expectations</li>
                </ol>
                """,
                "success"
            ),
            unsafe_allow_html=True
        )
with col3:
    st.markdown(
            Components.insight_box(
                "For Risk Analysts:",
                """
                <ol style="margin: 0; padding-left: 20px;">
                    <li>VaR (95%) quantifies daily loss exposure</li>
                    <li>Volatility clustering indicates when to reduce leverage</li>
                    <li>Drawdown analysis shows recovery patterns</li>
                    <li>Correlation matrix aids in diversification decisions</li>
                </ol>
                """,
                "error"
            ),
            unsafe_allow_html=True
        )
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
