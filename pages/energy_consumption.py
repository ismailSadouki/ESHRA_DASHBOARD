import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Utility Consumption", layout="wide")
st.title("📊 Utility Consumption Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("building_consumption.csv", parse_dates=["timestamp"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.to_period("M").dt.to_timestamp()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.date
    df['weekday'] = df['timestamp'].dt.day_name()
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
campuses = st.sidebar.multiselect(
    "Select Building ID(s)",
    df['campus_id'].unique(),
    default=df['campus_id'].unique()
)
df = df[df['campus_id'].isin(campuses)]

# Forecasting function
def forecast_utility(df):
    st.subheader("🔮 Utility Consumption Forecast (Next 30 Days)")

    # Prepare the data for Prophet
    df_daily = df.groupby('day')['consumption'].sum().reset_index()
    df_daily.columns = ['ds', 'y']  # Prophet expects 'ds' and 'y' column names

    # Fit the Prophet model
    model = Prophet()

    # Fit the model to the historical data
    model.fit(df_daily)

    # Manually create the future dates (next 30 days)
    last_date = df_daily['ds'].max()
    future_dates = pd.date_range(last_date, periods=31, freq='D')[1:]  # Exclude the last date (already present)

    # Prepare the dataframe with future dates
    future_df = pd.DataFrame({'ds': future_dates})

    # Make the forecast
    forecast = model.predict(future_df)

    # Plot the forecast with zoomed-in view for the next 30 days
    fig = go.Figure()

    # Plot the historical data
    fig.add_trace(go.Scatter(x=df_daily['ds'], y=df_daily['y'], mode='lines', name='Historical Consumption', line=dict(color='blue', width=2)))

    # Plot the forecasted data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Consumption', line=dict(dash='dash', color='orange', width=2)))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Confidence Interval', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0,100,80,0.2)'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Confidence Interval', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0,100,80,0.2)'
    ))

    # Update layout with zoomed-in effect
    fig.update_layout(
        title="Utility Consumption Forecast for Next 30 Days",
        xaxis_title="Date",
        yaxis_title="Consumption",
        template="plotly_dark",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            tickformat="%Y-%m-%d",
            range=[str(df_daily['ds'].max() - pd.Timedelta(days=7)), str(forecast['ds'].max())]  # Zoom in on the forecast period
        ),
        yaxis=dict(range=[df_daily['y'].min() * 0.9, forecast['yhat_upper'].max() * 1.1]),  # Adjust y-axis range to include forecast confidence
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


# Display the forecast
forecast_utility(df)

# Plot 1: Daily Total Consumption
daily_total = df.groupby('day')['consumption'].sum().reset_index()
fig1 = px.line(daily_total, x='day', y='consumption', title="📈 Daily Total Consumption")
fig1.update_layout(xaxis_title="Date", yaxis_title="Consumption")

# Plot 2: Total per campus
campus_totals = df.groupby('campus_id')['consumption'].sum().reset_index()
fig2 = px.bar(
    campus_totals,
    x='campus_id',
    y='consumption',
    labels={'campus_id': 'Building Id', 'consumption': 'Total Consumption'},
    title="🏢 Total Consumption by Building"
)

# Plot 3: Consumption Over Time for Selected Building
st.markdown("### 🔎 Breakdown per Building")
selected = st.selectbox("Choose a building:", campus_totals['campus_id'])
df_subset = df.tail(1000)
campus_data = df_subset[df_subset['campus_id'] == selected]
fig3 = px.line(campus_data, x='timestamp', y='consumption', title=f"📍 {selected} Consumption Over Time")

# Plot 4: Monthly Trend
monthly = df.groupby(['month', 'campus_id'])['consumption'].sum().reset_index()
fig4 = px.line(
    monthly,
    x='month',
    y='consumption',
    color='campus_id',
    labels={'campus_id': 'Building Id'},
    title="📅 Monthly Consumption per Building"
)
fig4.update_layout(xaxis_title="Month", yaxis_title="Consumption")

# Plot 5: Heatmap
heatmap_data = df.groupby(['weekday', 'hour'])['consumption'].mean().unstack().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)
fig5 = px.imshow(
    heatmap_data,
    title="🕓 Average Hourly Consumption by Day",
    labels={'x': 'Hour', 'y': 'Weekday'},
    color_continuous_scale='Viridis'
)

# Plot 6: Last 7 Days
recent = df[df['timestamp'] >= df['timestamp'].max() - pd.Timedelta(days=7)]
daily_recent = recent.groupby(recent['timestamp'].dt.date)['consumption'].sum().reset_index()
fig6 = px.bar(daily_recent, x='timestamp', y='consumption', title="📊 Last 7 Days Consumption")
fig6.update_layout(xaxis_title="Date", yaxis_title="Consumption")

# ==== SIDE-BY-SIDE DISPLAY ====
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig3, use_container_width=True)
with col4:
    st.plotly_chart(fig4, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(fig5, use_container_width=True)
with col6:
    st.plotly_chart(fig6, use_container_width=True)

#st.markdown("<br><br><hr><p style='text-align:left;'>Developed by Ismail Sadouki ❤️</p>", unsafe_allow_html=True)
