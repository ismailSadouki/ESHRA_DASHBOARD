import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Set up the page
st.set_page_config(page_title="University Utilities Consumption Dashboard", layout="wide")
st.title("🔥 Gas Consumption Analysis")

# Load building consumption data (for electricity and water)
@st.cache_data
def load_building_data():
    df = pd.read_csv("building_consumption.csv", parse_dates=["timestamp"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.to_period("M").dt.to_timestamp()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.date
    df['weekday'] = df['timestamp'].dt.day_name()
    df['utility_type'] = 'electricity'
    return df

# Load gas consumption data
@st.cache_data
def load_gas_data():
    df_gas = pd.read_csv("gas_consumption.csv", parse_dates=["timestamp"])
    df_gas['timestamp'] = pd.to_datetime(df_gas['timestamp'])
    df_gas['year'] = df_gas['timestamp'].dt.year
    df_gas['month'] = df_gas['timestamp'].dt.to_period("M").dt.to_timestamp()
    df_gas['hour'] = df_gas['timestamp'].dt.hour
    df_gas['day'] = df_gas['timestamp'].dt.date
    df_gas['weekday'] = df_gas['timestamp'].dt.day_name()
    df_gas['utility_type'] = 'gas'
    return df_gas

# Load datasets
df_building = load_building_data()
df_gas = load_gas_data()

# Sidebar filters
campuses_gas = st.sidebar.multiselect("Select Building ID(s) for Gas Data", df_gas['campus_id'].unique(), default=df_gas['campus_id'].unique())
df_gas = df_gas[df_gas['campus_id'].isin(campuses_gas)]

# --- First row: Daily and Campus Total ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Daily Total Gas Consumption")
    daily_gas_total = df_gas.groupby('day')['consumption'].sum()
    fig1 = px.line(daily_gas_total, x=daily_gas_total.index, y='consumption', title="Daily Total Gas Consumption")
    fig1.update_layout(xaxis_title="Date", yaxis_title="Gas Consumption")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🏢 Total Gas Consumption by Building")
    campus_gas_total = df_gas.groupby('campus_id')['consumption'].sum()
    fig2 = px.bar(campus_gas_total, x=campus_gas_total.index, y='consumption', title="Total Gas Consumption per Building")
    fig2.update_layout(xaxis_title="Campus ID", yaxis_title="Gas Consumption")
    st.plotly_chart(fig2, use_container_width=True)

# --- Second row: Monthly and Hourly Trends ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("📆 Monthly Gas Consumption Trend")
    monthly_gas = df_gas.groupby(['month', 'campus_id'])['consumption'].sum().reset_index()
    fig3 = px.line(monthly_gas, x='month', y='consumption', color='campus_id', title="Monthly Gas Consumption per Campus")
    fig3.update_layout(xaxis_title="Month", yaxis_title="Gas Consumption")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("🕓 Hourly Gas Consumption Pattern")
    hourly_gas = df_gas.groupby(['hour'])['consumption'].mean()
    fig4 = px.line(hourly_gas, x=hourly_gas.index, y='consumption', title="Average Hourly Gas Consumption")
    fig4.update_layout(xaxis_title="Hour", yaxis_title="Gas Consumption")
    st.plotly_chart(fig4, use_container_width=True)

# --- Peak vs Off-Peak Analysis ---
st.subheader("⏰ Peak vs Off-Peak Consumption Comparison")

# Define peak hours (e.g., 6 AM - 9 AM and 5 PM - 9 PM)
peak_hours = list(range(6, 10)) + list(range(17, 22))

# Filter peak and off-peak consumption
peak_data = df_gas[df_gas['hour'].isin(peak_hours)]
off_peak_data = df_gas[~df_gas['hour'].isin(peak_hours)]

# Group by day and compare
daily_peak = peak_data.groupby('day')['consumption'].sum()
daily_off_peak = off_peak_data.groupby('day')['consumption'].sum()

comparison_df = pd.DataFrame({
    "Peak Consumption": daily_peak,
    "Off-Peak Consumption": daily_off_peak
}).fillna(0)

fig5 = px.line(comparison_df, x=comparison_df.index, y=["Peak Consumption", "Off-Peak Consumption"],
               labels={"value": "Consumption (Units)", "variable": "Consumption Type"},
               title="Peak vs Off-Peak Consumption Comparison")
st.plotly_chart(fig5, use_container_width=True)

# Summary
peak_vs_off_peak_ratio = daily_peak.sum() / daily_off_peak.sum()
st.markdown("### Summary: Peak vs Off-Peak Consumption")
st.markdown(f"**Total Peak Consumption:** {daily_peak.sum():,.2f} units")
st.markdown(f"**Total Off-Peak Consumption:** {daily_off_peak.sum():,.2f} units")
st.markdown(f"**Peak to Off-Peak Ratio:** {peak_vs_off_peak_ratio:.2f}")


#st.markdown("<br><br><hr><p style='text-align:left;'>Developed by Ismail Sadouki ❤️</p>", unsafe_allow_html=True)
