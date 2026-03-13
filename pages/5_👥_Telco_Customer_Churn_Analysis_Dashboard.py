import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  
import joblib  
import seaborn as sns  
import matplotlib.pyplot as plt 

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Telco Customer Churn Analysis Dashboard", "👥")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')

    # Clean TotalCharges  
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)

    # Encode Churn as binary 
    df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

@st.cache_resource  
def load_model(): 
    try:
        model = joblib.load('churn_prediction_model.pkl')
        features = joblib.load('model_features.pkl') 
        return model, features 
    except:
        return None, None

df = load_data()  
model, model_features = load_model() 


# Title
st.markdown(
    Components.page_header(
        "👥 Telco Customer Churn Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("   ") 
st.subheader("🏠 :orange[Overview]", divider="orange")
st.markdown("   ") 
# Key Metrics Row  
col1, col2, col3, col4 = st.columns(4)  

total_customers = len(df)
churned_customers = df['Churn_Binary'].sum()
churn_rate = (churned_customers / total_customers) * 100
avg_revenue = df['MonthlyCharges'].mean()

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Customers",
            value=f"{total_customers:,}",
            delta="🔝",
            card_type="info",
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Churned Customers",
            value=f"{churned_customers:,}",
            delta=f"-{churn_rate:.1f}%",
            card_type="error"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Avg Monthly Revenue",
            value=f"${avg_revenue:.2f}",
            delta="💲",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col4:
    revenue_lost = df[df['Churn_Binary']==1]['MonthlyCharges'].sum()
    st.markdown(
        Components.metric_card(
            title="Monthly Revenue Lost",
            value=f"${revenue_lost:,.0f}",
            delta="❗",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("   ")
st.subheader("📉 :orange[Churn Distribution]")
churn_counts = df['Churn'].value_counts()
fig = px.pie(
    values=churn_counts.values,
    names=['Retained', 'Churned'],
    color_discrete_sequence=['#66c2a5', '#fc8d62'],
    hole=0.4
)
fig.update_layout(height=400)
st.plotly_chart(fig, width="stretch")

st.markdown("   ")
st.subheader("📊 :orange[Customer Distribution by Contract Type]")
contract_data = df['Contract'].value_counts()
fig2 = px.bar(
    x=contract_data.index,
    y=contract_data.values, 
    labels={'x': 'Contract Type', 'y': 'Number of Customers'},
    color=contract_data.values, 
    color_continuous_scale='Blues'
)
fig2.update_layout(height=400, showlegend=False) 
st.plotly_chart(fig2, width="stretch")  

# Tenure vs Monthly Charges Scatter  
st.markdown("   ")
st.subheader("📈 :orange[Tenure vs Monthly Charges by Churn Status]")
fig3 = px.scatter(
    df,
    x='tenure',
    y='MonthlyCharges',
    color='Churn',
    color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'},
    opacity=0.6, 
    labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)'},
    hover_data=['Contract', 'InternetService']
)
fig3.update_layout(height=500)
st.plotly_chart(fig3, width="stretch") 

st.subheader("📈 :blue[Churn Analysis]", divider="blue")
# Filter options  
col1, col2, col3 = st.columns(3) 
with col1:
    contract_filter = st.multiselect( 
        "Filter by Contract Type",
        options=df['Contract'].unique(), 
        default=df['Contract'].unique()  
    )
with col2:
    internet_filter = st.multiselect(
        "Filter by Internet Service",
        options=df['InternetService'].unique(),
        default=df['InternetService'].unique()
    )
with col3:
    tenure_range = st.slider(
        "Filter by Tenure (months)",
        min_value=int(df['tenure'].min()),
        max_value=int(df['tenure'].max()), 
        value=(int(df['tenure'].min()), int(df['tenure'].max()))
    ) 

# Apply filters  
filtered_df = df[ 
    (df['Contract'].isin(contract_filter)) & 
    (df['InternetService'].isin(internet_filter)) &  
    (df['tenure'] >= tenure_range[0]) &  
    (df['tenure'] <= tenure_range[1])  
]  
st.markdown(f"**Filtered Dataset: {len(filtered_df):,} customers**")  
st.markdown("   ")  

st.subheader("📊 :blue[Churn Rate by Contract Type]")  
contract_churn = filtered_df.groupby('Contract')['Churn_Binary'].mean() * 100 
fig4 = px.bar(
    x=contract_churn.index,
    y=contract_churn.values,
    labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
    color=contract_churn.values,
    color_continuous_scale='Reds',  
    text=contract_churn.values.round(1)  
)  
fig4.update_traces(texttemplate='%{text}%', textposition='outside')  
fig4.update_layout(height=400, showlegend=False) 
st.plotly_chart(fig4, width="stretch") 

st.subheader("📊 :blue[Churn Rate by Payment Method]")  
payment_churn = filtered_df.groupby('PaymentMethod')['Churn_Binary'].mean() * 100 
fig5 = px.bar(
    x=payment_churn.values,
    y=payment_churn.index,
    orientation='h', 
    labels={'x': 'Churn Rate (%)', 'y': 'Payment Method'},
    color=payment_churn.values, 
    color_continuous_scale='OrRd',
    text=payment_churn.values.round(1)  
)
fig5.update_traces(texttemplate='%{text}%', textposition='outside')
fig5.update_layout(height=400, showlegend=False) 
st.plotly_chart(fig5, width="stretch")

# Tenure Distribution  
st.subheader("📉 :blue[Tenure Distribution by Churn Status]")  
fig6 = go.Figure() 
fig6.add_trace(go.Histogram(  
    x=filtered_df[filtered_df['Churn']=='No']['tenure'],
    name='Retained', 
    marker_color='#3498db',
    opacity=0.7,
    nbinsx=30  
))
fig6.add_trace(go.Histogram(
    x=filtered_df[filtered_df['Churn']=='Yes']['tenure'],
    name='Churned', 
    marker_color='#e74c3c',  
    opacity=0.7,
    nbinsx=30  
)) 
fig6.update_layout( 
    barmode='overlay',
    xaxis_title='Tenure (months)',
    yaxis_title='Count',
    height=450, 
    legend=dict(x=0.7, y=0.95)  
)
st.plotly_chart(fig6, width="stretch")

# Churn by Services  
st.subheader("🔍 :blue[Churn Rate by Service Usage]") 
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'] 
service_churn_data = []  
for service in services:
    churn_rate = filtered_df.groupby(service)['Churn_Binary'].mean() * 100
for status, rate in churn_rate.items():
    service_churn_data.append({'Service': service, 'Status': status, 'Churn_Rate': rate  
}) 
service_df = pd.DataFrame(service_churn_data)

fig7 = px.bar( 
    service_df,
    x='Service',
    y='Churn_Rate',
    color='Status',
    barmode='group',
    labels={'Churn_Rate': 'Churn Rate (%)', 'Service': 'Service Type'},
    color_discrete_map={'Yes': '#27ae60', 'No': '#e74c3c', 'No internet service': '#95a5a6'}  
) 
fig7.update_layout(height=450)
st.plotly_chart(fig7, width="stretch")

# Key Insights Box  
st.markdown("   ")  
st.subheader("💡 :blue[Key Insights]")
col1, col2, col3 = st.columns(3)  
  
with col1:
    avg_tenure_churned = filtered_df[filtered_df['Churn']=='Yes']['tenure'].mean()  
    avg_tenure_retained = filtered_df[filtered_df['Churn']=='No']['tenure'].mean()  
    st.info(f"**Tenure Impact:**\n\n"
    f"Churned: {avg_tenure_churned:.1f} months\n\n" 
    f"Retained: {avg_tenure_retained:.1f} months")
with col2: 
    monthly_churned = filtered_df[filtered_df['Churn']=='Yes']['MonthlyCharges'].mean() 
    monthly_retained = filtered_df[filtered_df['Churn']=='No']['MonthlyCharges'].mean()  
    st.warning(f"**Monthly Charges:**\n\n"
    f"Churned: ${monthly_churned:.2f}\n\n" 
    f"Retained: ${monthly_retained:.2f}")
with col3:
    high_risk = filtered_df[(filtered_df['Contract'] == 'Month-to-month') & (filtered_df['tenure'] < 12)]  
    high_risk_rate = high_risk['Churn_Binary'].mean() * 100
    st.error(f"**High-Risk Segment:**\n\n"
    f"Month-to-month + Tenure < 12\n\n"
    f"Churn Rate: {high_risk_rate:.1f}%")

st.subheader("💰 :yellow[Revenue Impact]", divider="yellow")

# Calculate revenue metrics  
total_revenue = df['TotalCharges'].sum()  
churned_revenue = df[df['Churn']=='Yes']['TotalCharges'].sum()  
monthly_recurring_lost = df[df['Churn']=='Yes']['MonthlyCharges'].sum()  
avg_customer_value = df['TotalCharges'].mean()  

# Top Metrics  
col1, col2, col3, col4 = st.columns(4)  
with col1:
    st.markdown(
        Components.metric_card(
            title="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta="💲",
            card_type="success",
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Revenue Lost (Churn)",
            value=f"${churned_revenue:,.0f}",
            delta=f"-{(churned_revenue/total_revenue)*100:.1f}%",
            card_type="error"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Monthly Revenue Lost",
            value=f"${monthly_recurring_lost:,.0f}",
            delta="❗",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col4:
    revenue_lost = df[df['Churn_Binary']==1]['MonthlyCharges'].sum()
    st.markdown(
        Components.metric_card(
            title="Avg Customer Value",
            value=f"${avg_customer_value:.2f}",
            delta="🔝",
            card_type="info"
        ), unsafe_allow_html=True
    )
st.markdown("   ")  
st.subheader("📊 :yellow[Revenue Distribution by Contract Type]")
contract_revenue = df.groupby(['Contract', 'Churn'])['TotalCharges'].sum().reset_index() 
fig8 = px.bar(
    contract_revenue, 
    x='Contract',
    y='TotalCharges',
    color='Churn',
    barmode='group',
    labels={'TotalCharges': 'Total Revenue ($)', 'Contract': 'Contract Type'}, 
    color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'},
    text_auto='.2s'  
) 
fig8.update_layout(height=400)
st.plotly_chart(fig8, width="stretch")

st.subheader("📈 :yellow[Monthly Charges Distribution]")  
fig9 = go.Figure()  
fig9.add_trace(go.Box( 
    y=df[df['Churn']=='No']['MonthlyCharges'],
    name='Retained', 
    marker_color='#3498db'  
)) 
fig9.add_trace(go.Box( 
    y=df[df['Churn']=='Yes']['MonthlyCharges'],
    name='Churned',
    marker_color='#e74c3c'  
))  
fig9.update_layout(
    yaxis_title='Monthly Charges ($)',
    height=400  
)  
st.plotly_chart(fig9, width="stretch")

st.subheader("📉 :yellow[Projected Revenue Loss Over Time]")
months = st.slider("Select projection period (months)", 1, 36, 12) 
# Calculate projected loss  
monthly_loss_data = []  
cumulative_loss = 0 
for month in range(1, months + 1): 
    monthly_loss = monthly_recurring_lost 
    cumulative_loss += monthly_loss
    monthly_loss_data.append({
        'Month': month, 
        'Monthly Loss': monthly_loss, 
        'Cumulative Loss': cumulative_loss
    })
    loss_df = pd.DataFrame(monthly_loss_data) 
    fig10 = make_subplots(specs=[[{"secondary_y": True}]])
    fig10.add_trace( 
        go.Bar(x=loss_df['Month'], y=loss_df['Monthly Loss'], 
        name='Monthly Loss', marker_color='#e74c3c'),
        secondary_y=False 
    )
    fig10.add_trace(  
        go.Scatter(x=loss_df['Month'], y=loss_df['Cumulative Loss'], 
        name='Cumulative Loss', marker_color='#c0392b', 
        line=dict(width=3)),
        secondary_y=True
    )
    fig10.update_xaxes(title_text="Month") 
    fig10.update_yaxes(title_text="Monthly Loss ($)", secondary_y=False)
    fig10.update_yaxes(title_text="Cumulative Loss ($)", secondary_y=True)
    fig10.update_layout(height=500)
    st.plotly_chart(fig10, width="stretch")

st.info(f"💡 **Projection Insight:** If churn continues at current rate, "  
f"projected revenue loss over {months} months: **${cumulative_loss:,.0f}**")  

st.markdown("   ")  
st.subheader("💎 :yellow[Customer Lifetime Value (CLV) Analysis]")  
# CLV by Contract Type  
clv_contract = df.groupby('Contract').agg({
    'TotalCharges': 'mean',
    'tenure': 'mean', 
    'MonthlyCharges': 'mean' 
}).reset_index()
fig11 = px.bar(
    clv_contract,
    x='Contract',
    y='TotalCharges', 
    labels={'TotalCharges': 'Average CLV ($)', 'Contract': 'Contract Type'},
    color='TotalCharges',
    color_continuous_scale='Greens', 
    text='TotalCharges'  
)  
fig11.update_traces(texttemplate='$%{text:.0f}', textposition='outside')  
fig11.update_layout(height=400, showlegend=False) 
st.plotly_chart(fig11, width="stretch")

# CLV by Churn Status  
clv_churn = df.groupby('Churn').agg({ 
    'TotalCharges': 'mean',
    'tenure': 'mean',
    'MonthlyCharges': 'mean' 
}).reset_index() 
fig12 = px.bar(  
    clv_churn,  
    x='Churn',  
    y='TotalCharges',  
    labels={'TotalCharges': 'Average CLV ($)', 'Churn': 'Customer Status'},  
    color='Churn',  
    color_discrete_map={'Yes': '#e74c3c', 'No': '#27ae60'},  
    text='TotalCharges'  
)  
fig12.update_traces(texttemplate='$%{text:.0f}', textposition='outside')  
fig12.update_layout(height=400, showlegend=False) 
st.plotly_chart(fig12, width="stretch")

# Top Revenue Customers at Risk  
st.markdown("   ")  
st.subheader("⚠️ :yellow[High-Value Customers at Risk]")
at_risk_customers = df[
    (df['Churn'] == 'Yes') &  
    (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75))  
].sort_values('TotalCharges', ascending=False).head(10) 

st.dataframe( 
    at_risk_customers[['customerID', 'Contract', 'tenure', 'MonthlyCharges',
    'TotalCharges', 'PaymentMethod']].style.format({ 
        'MonthlyCharges': '${:.2f}', 
        'TotalCharges': '${:.2f}' 
    }),
    width="stretch"
)
total_at_risk_revenue = at_risk_customers['MonthlyCharges'].sum()  
st.warning(f"🚨 **Alert:** {len(at_risk_customers)} high-value customers churned, "  
f"resulting in **${total_at_risk_revenue:,.2f}/month** revenue loss!") 

st.subheader("💻 :violet[Churn Predictor]", divider = "violet")
if model is None:
    st.error("⚠️ Model not found! Please train the model first.")
    st.info("Run the training script to generate 'churn_prediction_model.pkl'")
else:
    st.success("✅ Model loaded successfully!") 

st.markdown("   ")  
st.subheader("📝 :violet[Enter Customer Information]")

# Create input form  
col1, col2 = st.columns(2)  
  
with col1:  
    st.markdown("**Demographics**")  
    gender = st.selectbox("Gender", ["Male", "Female"])  
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])  
    partner = st.selectbox("Has Partner", ["No", "Yes"])  
    dependents = st.selectbox("Has Dependents", ["No", "Yes"]) 
with col2:
    st.markdown("**Account Information**")
    tenure = st.slider("Tenure (months)", 0, 72, 12) 
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])  
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])  
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
    "Bank transfer (automatic)", "Credit card (automatic)"])

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Services**")
    phone = st.selectbox("Phone Service", ["No", "Yes"])  
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])  
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]) 
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"]) 
with col4:
    st.markdown("**Services**")
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])  
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])  
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"]) 
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])  
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"]) 

col5, col6 = st.columns(2)  
with col5: 
    monthly_charges = st.number_input("Monthly Charges ($)", 
    min_value=0.0, max_value=150.0,  
    value=70.0, step=0.5)  
with col6:  
    total_charges = st.number_input("Total Charges ($)", 
    min_value=0.0, max_value=10000.0, 
    value=monthly_charges * tenure, step=10.0) 

st.markdown("# :red[Click to predict Churn]")
if st.button("🔮 Predict Churn Probability", type="primary", width="content"):
        # Prepare input data
        input_data = pd.DataFrame({  
            'SeniorCitizen': [1 if senior == "Yes" else 0],  
            'Partner': [1 if partner == "Yes" else 0],
            'Dependents': [1 if dependents == "Yes" else 0],
            'tenure': [tenure],  
            'PhoneService': [1 if phone == "Yes" else 0],  
            'PaperlessBilling': [1 if paperless == "Yes" else 0], 
            'MonthlyCharges': [monthly_charges],  
            'TotalCharges': [total_charges],  
            'gender': [1 if gender == "Male" else 0] 
        })
        # Add categorical features (one-hot encoded)  
        # This is simplified - in production, use the exact preprocessing pipeline
        categorical_mapping = {
            'MultipleLines': multiple_lines,  
            'InternetService': internet,  
            'OnlineSecurity': online_security,  
            'OnlineBackup': online_backup,  
            'DeviceProtection': device_protection,  
            'TechSupport': tech_support,  
            'StreamingTV': streaming_tv,  
            'StreamingMovies': streaming_movies,  
            'Contract': contract,  
            'PaymentMethod': payment  
        }
        # Create dummy variables  
        for model_feature in model_features:
            if model_feature not in input_data.columns:
                found = False
                for feature, value in categorical_mapping.items():
                    if model_feature.startswith(feature + "_"):
                        expected_value = model_feature.split("_", 1)[1]
                        input_data[model_feature] = [1 if value == expected_value else 0]
                        found = True
                        break
                if not found:
                    input_data[model_feature] = 0
        input_data = input_data[model_features]
        churn_probability = model.predict_proba(input_data)[0][1]
        churn_prediction = model.predict(input_data)[0]

        # Display results  
        st.markdown("   ") 
        st.subheader("🎯 :violet[Prediction Results]")
        col1, col2, col3 = st.columns(3) 
        with col1:  
            st.metric("Churn Probability", f"{churn_probability*100:.1f}%") 

        with col2:  
            prediction_label = "🔴 LIKELY TO CHURN" if churn_prediction == 1 else "🟢 LIKELY TO STAY"  
            st.metric("Prediction", prediction_label)  
        with col3:  
            if churn_probability > 0.7:
                risk_level = 'HIGH'
                risk_color = "🔴"
            elif churn_probability > 0.4:
                risk_level = "MEDIUM"
                risk_color = "🟡"
            else:
                risk_level = "LOW"
                risk_color = "🟢"
            st.metric("Risk Level", f"{risk_color} {risk_level}")  

        # Probability gauge  
        fig13 = go.Figure(go.Indicator( 
            mode="gauge+number+delta", 
            value=churn_probability * 100,  
            domain={'x': [0, 1], 'y': [0, 1]}, 
            title={'text': "Churn Probability (%)", 'font': {'size': 24}},  
            delta={'reference': 50, 'increasing': {'color': "red"}},  
            gauge={ 
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"}, 
                'bar': {'color': "darkblue"},  
                'bgcolor': "white", 
                'borderwidth': 2, 
                'bordercolor': "gray",  
                'steps': [  
                    {'range': [0, 30], 'color': '#27ae60'},  
                    {'range': [30, 70], 'color': '#f39c12'},  
                    {'range': [70, 100], 'color': '#e74c3c'}  
                ],
                'threshold': {  
                    'line': {'color': "red", 'width': 4},  
                    'thickness': 0.75, 
                    'value': 50  
                }
            }
        ))
        fig13.update_layout(height=400) 
        st.plotly_chart(fig13, width="stretch")
        # Recommendations  
        st.markdown("   ")  
        st.subheader("💡 :violet[Retention Recommendations]")  
        if churn_probability > 0.7:  
            st.error("**🚨 HIGH RISK - Immediate Action Required!**")  
            recommendations = [  
                "📞 **Priority Contact:** Reach out within 24 hours with personalized retention offer",  
                "💰 **Special Discount:** Offer 20-30% discount for upgrading to annual contract",  
                "🎁 **Value-Added Services:** Provide free premium services (tech support, security) for 3 months",  
                "📋 **Account Review:** Schedule dedicated account manager meeting",  
                "🔒 **Contract Upgrade:** Incentivize switch from month-to-month to long-term contract"  
            ]
        elif churn_probability > 0.4:  
            st.warning("**⚠️ MEDIUM RISK - Proactive Engagement Needed**")  
            recommendations = [  
                "📧 **Email Campaign:** Send personalized service upgrade offers",  
                "🎯 **Targeted Promotion:** 10-15% discount on annual plans", 
                "📞 **Customer Survey:** Understand pain points and satisfaction levels",  
                "🛠️ **Service Enhancement:** Recommend additional services based on usage patterns",  
                "💳 **PaymentMethod:** Encourage automatic payment methods with small discount"  
            ]
        else:
            st.success("**✅ LOW RISK - Maintain Engagement**") 
            recommendations = [  
                "🌟 **Loyalty Program:** Enroll in rewards program for long-term customers",  
                "📨 **Regular Communication:** Monthly newsletter with tips and new features",  
                "🎁 **Appreciation Gestures:** Occasional bonus services or credits",  
                "📊 **Usage Insights:** Provide personalized usage reports and optimization tips",  
                "🔄 **Cross-Sell:** Recommend complementary services based on current usage"  
            ]
        for i, rec in enumerate(recommendations, 1):  
            st.markdown(f"{i}. {rec}")  

# Financial Impact  
st.markdown("   ")  
st.subheader("💰 :violet[Financial Impact Analysis]") 

col1, col2 = st.columns(2)  
  
with col1:  
    monthly_loss = monthly_charges  
    annual_loss = monthly_charges * 12  
    lifetime_loss = monthly_charges * 24 # Assume 2-year average lifetime
    st.info(f"**Potential Revenue Loss if Customer Churns:**\n\n" 
    f"• Monthly: ${monthly_loss:.2f}\n\n" 
    f"• Annual: ${annual_loss:.2f}\n\n"  
    f"• Lifetime (24 months): ${lifetime_loss:.2f}")
with col2:  
    retention_cost = monthly_charges * 0.2 # Assume 20% discount for retention 
    roi = (monthly_charges * 12) - retention_cost 
    st.success(f"**Retention Investment ROI:**\n\n"  
    f"• Retention Cost (20% discount): ${retention_cost:.2f}\n\n" 
    f"• Annual Retention Value: ${monthly_charges * 12:.2f}\n\n"  
    f"• Net ROI: ${roi:.2f} ({(roi/(retention_cost))*100:.0f}% return)") 

st.subheader("📊 :rainbow[Customer Segmentation]", divider="rainbow")
# Create segments  
df['Tenure_Segment'] = pd.cut(df['tenure'], 
bins=[0, 12, 24, 48, 100],  
labels=['0-12 months', '12-24 months', '24-48 months', '48+ months']) 

df['Revenue_Segment'] = pd.cut(df['MonthlyCharges'],
bins=[0, 35, 70, 100, 150], 
labels=['Low ($0-35)', 'Medium ($35-70)', 'High ($70-100)', 'Premium ($100+)']) 

# Segment Summary  
segment_summary = df.groupby([
    'Contract',
    'Tenure_Segment']).agg({ 
        'customerID': 'count', 
        'Churn_Binary': 'mean',
        'MonthlyCharges': 'mean', 
        'TotalCharges': 'sum' 
    }).reset_index()
segment_summary.columns = ['Contract', 'Tenure', 'Customers', 'Churn_Rate', 'Avg_Monthly', 'Total_Revenue'] 
segment_summary['Churn_Rate'] = segment_summary['Churn_Rate'] * 100  

# Heatmap  
pivot_churn = segment_summary.pivot(index='Tenure', columns='Contract', values='Churn_Rate') 

fig14 = px.imshow(pivot_churn,
labels=dict(x="Contract Type", y="Tenure Segment", color="Churn Rate (%)"),  
x=pivot_churn.columns,  
y=pivot_churn.index,  
color_continuous_scale='RdYlGn_r',  
text_auto='.1f',  
aspect="auto")  

fig14.update_layout(height=400, title="Churn Rate Heatmap by Segment") 
st.plotly_chart(fig14, width="stretch") 

  
# Customer Count by Segment  
st.subheader("👥 :rainbow[Customer Distribution by Segment]") 

pivot_customers = segment_summary.pivot(index='Tenure', columns='Contract', values='Customers') 
fig15 = px.bar(segment_summary,  
x='Tenure',  
y='Customers',  
color='Contract',  
barmode='group',  
labels={'Customers': 'Number of Customers'},  
color_discrete_sequence=px.colors.qualitative.Set2)  
fig15.update_layout(height=400)
st.plotly_chart(fig15, width="stretch")

# Revenue by segment  
fig16 = px.bar(segment_summary,
x='Tenure',  
y='Total_Revenue',  
color='Contract',  
barmode='group',  
labels={'Total_Revenue': 'Total Revenue ($)'},  
color_discrete_sequence=px.colors.qualitative.Pastel)  
fig16.update_layout(height=400) 
st.plotly_chart(fig16, width="stretch")

# Revenue Segmentation  
st.markdown("   ")  
st.subheader("💵 :rainbow[Revenue-Based Segmentation]") 
revenue_analysis = df.groupby(['Revenue_Segment', 'Churn']).agg({ 
    'customerID': 'count',
    'MonthlyCharges': 'sum'  
}).reset_index()  
revenue_analysis.columns = ['Revenue_Segment', 'Churn', 'Customers', 'Total_Monthly_Revenue']  
fig17 = px.sunburst(revenue_analysis,  
path=['Revenue_Segment', 'Churn'],  
values='Customers',  
color='Total_Monthly_Revenue',  
color_continuous_scale='Blues',  
title='Customer Distribution: Revenue Segments & Churn Status')  
fig17.update_layout(height=600)  
st.plotly_chart(fig17, width="stretch")

# Service Bundle Analysis  
st.markdown("   ")  
st.subheader("📦 :rainbow[Service Bundle Analysis]")  

# Count number of services per customer  
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']  
df['Service_Count'] = df[service_cols].apply( 
    lambda row: sum([1 for x in row if x not in ['No', 'No phone service', 'No internet service']]), axis=1)

service_bundle = df.groupby('Service_Count').agg({ 
    'customerID': 'count',
    'Churn_Binary': 'mean',
    'MonthlyCharges': 'mean'
}).reset_index()
service_bundle.columns = ['Services', 'Customers', 'Churn_Rate', 'Avg_Monthly_Charges']  
service_bundle['Churn_Rate'] = service_bundle['Churn_Rate']* 100  

fig18 = make_subplots(specs=[[{"secondary_y": True}]]) 
fig18.add_trace( 
    go.Bar(x=service_bundle['Services'], 
    y=service_bundle['Customers'],
    name='Number of Customers',
    marker_color='lightblue'), 
    secondary_y=False  
) 
fig18.add_trace( 
    go.Scatter(x=service_bundle['Services'],
    y=service_bundle['Churn_Rate'],
    name='Churn Rate (%)', 
    marker_color='red',
    line=dict(width=3)),
    secondary_y=True  
) 
fig18.update_xaxes(title_text="Number of Services")  
fig18.update_yaxes(title_text="Number of Customers", secondary_y=False)  
fig18.update_yaxes(title_text="Churn Rate (%)", secondary_y=True)  
fig18.update_layout(height=450, title="Customer Count & Churn Rate by Service Bundle Size") 
st.plotly_chart(fig18, width="stretch")

# Demographic Segmentation  
st.markdown("   ")  
st.subheader("👨‍👩‍👧‍👦 :red[Demographic Segmentation]", divider="red")
# Senior Citizen Analysis  
senior_analysis = df.groupby(['SeniorCitizen', 'Churn']).size().reset_index(name='Count') 
senior_analysis['SeniorCitizen'] = senior_analysis['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'}) 

fig19 = px.bar(
    senior_analysis,
    x='SeniorCitizen',
    y='Count',
    color='Churn',
    barmode='group',
    title='Churn by Senior Citizen Status',
    color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'}
)
fig19.update_layout(height=400)
st.plotly_chart(fig19, width="stretch")

# Partner/Dependents Analysis  
family_analysis = df.groupby(['Partner', 'Dependents', 'Churn']).size().reset_index(name='Count') 
family_analysis['Family_Status'] = family_analysis.apply(lambda x: f"Partner: {x['Partner']}, Dependents: {x['Dependents']}", axis=1) 

fig20 = px.bar(
    family_analysis,
    x='Family_Status',
    y='Count',
    color='Churn',
    barmode='group',
    title='Churn by Family Status',
    color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'}
)
fig20.update_xaxes(tickangle=45)
fig20.update_layout(height=400)
st.plotly_chart(fig20, width="stretch")

# Segment Performance Table  
st.markdown("   ")  
st.subheader("📋 :red[Detailed Segment Performance]")  

# Create comprehensive segment analysis 
segment_detail = df.groupby(['Contract', 'Tenure_Segment', 'Revenue_Segment']).agg({
    'customerID': 'count',
    'Churn_Binary': 'mean',
    'MonthlyCharges': 'mean', 
    'TotalCharges': 'mean',
    'tenure': 'mean'
}).reset_index()

segment_detail.columns = ['Contract', 'Tenure_Segment', 'Revenue_Segment', 'Customer_Count', 'Churn_Rate', 'Avg_Monthly_Charges', 'Avg_Total_Charges', 'Avg_Tenure']
segment_detail['Churn_Rate'] = (segment_detail['Churn_Rate'] * 100).round(2)  
segment_detail['Avg_Monthly_Charges'] = segment_detail['Avg_Monthly_Charges'].round(2) 
segment_detail['Avg_Total_Charges'] = segment_detail['Avg_Total_Charges'].round(2) 
segment_detail['Avg_Tenure'] = segment_detail['Avg_Tenure'].round(1)
# Sort by churn rate descending  
segment_detail = segment_detail.sort_values('Churn_Rate', ascending=False)  
# Display with color coding  
st.dataframe(segment_detail.head(20).style.background_gradient(
    subset=['Churn_Rate'], 
    cmap='RdYlGn_r')
    .background_gradient(
        subset=['Avg_Monthly_Charges', 'Avg_Total_Charges'],
        cmap='Blues')
        .format({
            'Churn_Rate': '{:.2f}%',
            'Avg_Monthly_Charges': '${:.2f}',
            'Avg_Total_Charges': '${:.2f}',
            'Avg_Tenure': '{:.1f} months'
        }),
        width="stretch"
    )
# Strategic Insights  
st.markdown("   ")  
st.subheader("🎯 :red[Strategic Segment Insights]")
col1, col2, col3 = st.columns(3)  
with col1:  
    # High-risk segment
    high_risk_segment = segment_detail.nlargest(1, 'Churn_Rate').iloc[0]
    st.error(f"**🔴 Highest Risk Segment:**\n\n"
    f"Contract: {high_risk_segment['Contract']}\n\n"
    f"Tenure: {high_risk_segment['Tenure_Segment']}\n\n"
    f"Revenue: {high_risk_segment['Revenue_Segment']}\n\n"
    f"Revenue: {high_risk_segment['Revenue_Segment']}\n\n"  
    f"Churn Rate: {high_risk_segment['Churn_Rate']:.1f}%\n\n"
    f"Customers: {int(high_risk_segment['Customer_Count'])}") 
with col2:
    # Most valuable segment 
    high_value_segment = segment_detail.nlargest(1, 'Avg_Total_Charges').iloc[0] 
    st.success(f"**💎 Most Valuable Segment:**\n\n"
    f"Contract: {high_value_segment['Contract']}\n\n" 
    f"Tenure: {high_value_segment['Tenure_Segment']}\n\n" 
    f"Revenue: {high_value_segment['Revenue_Segment']}\n\n"
    f"Avg CLV: ${high_value_segment['Avg_Total_Charges']:.0f}\n\n"
    f"Customers: {int(high_value_segment['Customer_Count'])}") 
with col3:
    # Best retention segment 
    best_retention = segment_detail.nsmallest(1, 'Churn_Rate').iloc[0]  
    st.info(f"**🟢 Best Retention Segment:**\n\n" 
    f"Contract: {best_retention['Contract']}\n\n"  
    f"Tenure: {best_retention['Tenure_Segment']}\n\n"  
    f"Revenue: {best_retention['Revenue_Segment']}\n\n"  
    f"Churn Rate: {best_retention['Churn_Rate']:.1f}%\n\n"  
    f"Customers: {int(best_retention['Customer_Count'])}") 

# Action Plan  
st.markdown("   ")  
st.subheader("📝 :red[Recommended Action Plan by Segment]")  
action_plans = { 
    "🔴 High-Risk Segments": [ 
        "**Month-to-month + Short Tenure (0-12 months):**", 
        " • Immediate outreach with contract upgrade incentives (25-30% discount)", 
        " • Assign dedicated retention specialists", 
        " • Offer 3-month trial of premium services", 
        " • Implement early warning monitoring system" 
    ],
    "🟡 Medium-Risk Segments": [
        "**One-year contracts nearing renewal:**",
        " • Proactive renewal campaigns 60 days before expiration",
        " • Loyalty rewards for contract extension", 
        " • Usage analysis and personalized recommendations", 
        " • Competitive comparison highlighting value" 
    ],
    "🟢 Low-Risk/High-Value Segments": [ 
        "**Two-year contracts + Long Tenure:**",
        " • VIP treatment and priority support", 
        " • Exclusive early access to new features",
        " • Referral incentive programs",
        " • Annual appreciation events/gifts"
    ]
}
for category, actions in action_plans.items():
    with st.expander(category, expanded=True):
        for action in actions:
            st.markdown(action)
st.markdown("   ")
st.subheader("📚 :green[Additional Analysis - Advanced Techniques]", divider="green")

st.subheader(":green[Customer Cohort Analysis]")
# Create cohort based on sign-up period 
df['Cohort'] = pd.cut(df['tenure'], 
bins=[0, 6, 12, 24, 36, 100],
labels=['0-6m', '6-12m', '12-24m', '24-36m', '36m+']) 

# Cohort retention analysis  
cohort_analysis = df.groupby('Cohort').agg({ 
    'Churn_Binary': ['count', 'sum', 'mean']
}).reset_index()

cohort_analysis.columns = ['Cohort', 'Total_Customers', 'Churned', 'Churn_Rate']
cohort_analysis['Retention_Rate'] = (1 - cohort_analysis['Churn_Rate']) * 100
cohort_analysis['Churn_Rate'] = cohort_analysis['Churn_Rate'] * 100

print("\nCohort Analysis:")
print(cohort_analysis)

# Visualize cohort retention
fig21 = go.Figure()
fig21.add_trace(go.Bar(
    x=cohort_analysis['Cohort'],
    y=cohort_analysis['Retention_Rate'],
    name='Retention Rate',
    marker_color='#27ae60',
    text=cohort_analysis['Retention_Rate'].round(1),
    texttemplate='%{text}%',
    textposition='outside'
))
fig21.add_trace(go.Bar(
    x=cohort_analysis['Cohort'],
    y=cohort_analysis['Churn_Rate'],
    name='Churn Rate',
    marker_color='#e74c3c',
    text=cohort_analysis['Churn_Rate'].round(1),
    texttemplate='%{text}%',
    textposition='outside'
))
fig21.update_layout(
    title='Cohort Retention vs Churn Rate',
    xaxis_title='Customer Cohort (Tenure)',
    yaxis_title='Rate (%)',
    barmode='stack',
    height=500
)
st.plotly_chart(fig21, width="stretch")

st.subheader(":green[RFM(Recency, Frequency, Monetary)Analysis]")
# RFM Analysis for customer segmentation  
# Recency: Tenure (how long they've been customers)  
# Frequency: Service adoption (number of services)  
# Monetary: MonthlyCharges 
df['R_Score'] = pd.qcut(df['tenure'], q=4, labels=[1, 2, 3, 4]) 
df['F_Score'] = pd.qcut(df['Service_Count'].rank(method='first'), q=4, labels=[1, 2, 3, 4])  
df['M_Score'] = pd.qcut(df['MonthlyCharges'], q=4, labels=[1, 2, 3, 4])

# Combine scores  
df['RFM_Score'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str) 
df['RFM_Total'] = df['R_Score'].astype(int) + df['F_Score'].astype(int) + df['M_Score'].astype(int) 

# Segment customers based on RFM 
def rfm_segment(row):
    if row['RFM_Total'] >= 9:
        return 'Champions'
    elif row['RFM_Total'] >= 7:
        return 'Loyal Customers'
    elif row['RFM_Total'] >= 5:
        return 'Potential Loyalists'
    elif row['RFM_Total'] >= 4:
        return 'At Risk'
    else:
        return 'Need Attention'
df['RFM_Segment'] = df.apply(rfm_segment, axis=1)

# Analyze churn by RFM segment
rfm_churn = df.groupby('RFM_Segment').agg({
    'customerID': 'count',
    'Churn_Binary': 'mean',
    'MonthlyCharges': 'mean',
    'TotalCharges': 'sum'
}).reset_index()

rfm_churn.columns = ['Segment', 'Customers', 'Churn_Rate', 'Avg_Monthly', 'Total_Revenue']
rfm_churn['Churn_Rate'] = rfm_churn['Churn_Rate'] * 100

print("\nRFM Segmentation Analysis:")
print(rfm_churn.sort_values('Churn_Rate', ascending=False))

fig22 = px.scatter(
    rfm_churn,
    x='Customers',
    y='Churn_Rate',
    size='Total_Revenue',
    color='Segment',
    hover_data=['Avg_Monthly'],
    title='RFM Segments: Customer Count vs Churn Rate (bubble size = revenue)',
    labels={'Churn_Rate': 'Churn Rate (%)'}
)
fig22.update_layout(height=500)
st.plotly_chart(fig22, width="stretch")

st.subheader(":green[Time Series Forecasting - Predict Future Churn]")
from sklearn.ensemble import GradientBoostingRegressor

# Create monthly aggregation
df['Month_Joined'] = df['tenure'].apply(lambda x: 72 - x)
# Approximate join month
monthly_churn = df.groupby('Month_Joined').agg({
    'Churn_Binary': 'mean',
    'customerID': 'count'
}).reset_index()

monthly_churn.columns = ['Month', 'Churn_Rate', 'New_Customers']
monthly_churn = monthly_churn.sort_values('Month')

# Simple forecasting model
from sklearn.linear_model import LinearRegression

X = monthly_churn['Month'].values.reshape(-1, 1)
y = monthly_churn['Churn_Rate'].values

# Fit model
lr_forecast = LinearRegression()
lr_forecast.fit(X, y)

# Predict next 12 months
future_months = np.arange(monthly_churn['Month'].max() + 1, monthly_churn['Month'].max() + 13).reshape(-1, 1)
future_churn = lr_forecast.predict(future_months)

fig23 = go.Figure()
fig23.add_trace(go.Scatter(
    x=monthly_churn['Month'],
    y=monthly_churn['Churn_Rate'] * 100,
    mode='lines+markers',
    name='Historical Churn Rate',
    line=dict(color='#3498db', width=3)
))
fig23.add_trace(go.Scatter(
    x=future_months.flatten(),
    y=future_churn * 100,
    mode='lines+markers',
    name='Forecasted Churn Rate',
    line=dict(color='#e74c3c', width=3, dash='dash')
))
fig23.update_layout(
    title='Churn Rate Forecast - Next 12 Months',
    xaxis_title='Month',
    yaxis_title='Churn Rate (%)',
    height=500,
    hovermode='x unified'
)
st.plotly_chart(fig23, width="stretch")