import streamlit as st
import pandas as pd
import plotly.express as px
<<<<<<< HEAD

# Load Data
@st.cache_data
def load_water_data():
    df = pd.read_csv("water_consumption.csv", parse_dates=["timestamp"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.to_period("M").dt.to_timestamp()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.date
    df['weekday'] = df['timestamp'].dt.day_name()
    return df

df_water = load_water_data()

# Filters
=======
from utils import DataManager


# Load Data
dm = DataManager()
df_water = dm.load_water()

>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.sidebar.header("💧 Water Filters")
campuses = st.sidebar.multiselect("Choose Campus ID(s):", df_water["campus_id"].unique(), default=df_water["campus_id"].unique())
df_water = df_water[df_water["campus_id"].isin(campuses)]

<<<<<<< HEAD
# Page Title
st.title("💧 Water Consumption Dashboard")

# --- Section 1: Daily Water Consumption
=======
st.title("💧 Water Consumption Dashboard")

>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.subheader("📈 Daily Water Consumption")
daily = df_water.groupby("day")["consumption"].sum().reset_index()
fig_daily = px.line(daily, x="day", y="consumption", title="Daily Water Consumption", labels={"day": "Date", "consumption": "Units"})
st.plotly_chart(fig_daily, use_container_width=True)

<<<<<<< HEAD
# --- Section 2: Peak vs Off-Peak Water Consumption
=======
>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.subheader("⏰ Peak vs Off-Peak Water Comparison")
peak_hours = list(range(6, 10)) + list(range(17, 22))
df_water["time_type"] = df_water["hour"].apply(lambda h: "Peak" if h in peak_hours else "Off-Peak")
time_compare = df_water.groupby(["day", "time_type"])["consumption"].sum().reset_index()
fig_compare = px.line(time_compare, x="day", y="consumption", color="time_type", title="Peak vs Off-Peak Water Consumption", labels={"day": "Date", "consumption": "Units", "time_type": "Time Type"})
st.plotly_chart(fig_compare, use_container_width=True)

<<<<<<< HEAD
# --- Section 3: Summary
=======
>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.subheader("📊 Summary: Peak vs Off-Peak Water")
total_peak = time_compare[time_compare["time_type"] == "Peak"]["consumption"].sum()
total_offpeak = time_compare[time_compare["time_type"] == "Off-Peak"]["consumption"].sum()
ratio = total_peak / total_offpeak if total_offpeak else float("inf")
st.markdown(f"**Total Peak Water Consumption:** {total_peak:.2f} units")
st.markdown(f"**Total Off-Peak Water Consumption:** {total_offpeak:.2f} units")
st.markdown(f"**Peak to Off-Peak Ratio:** {ratio:.2f}")






















df = df_water



<<<<<<< HEAD
# ---------- Page Setup ----------

# ---------- Title ----------
st.markdown("## 💧 Water Savings Opportunities")
st.markdown("---")

# ---------- Data Preprocessing ----------
=======
st.markdown("## 💧 Water Savings Opportunities")
st.markdown("---")

>>>>>>> 4cd54e7 (Initial commit after reinitializing)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.day_name()

<<<<<<< HEAD
# ---------- Charts in the Same Row ----------
=======
>>>>>>> 4cd54e7 (Initial commit after reinitializing)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⏱️ Hourly Water Usage Trend")
    hourly_avg = df.groupby("hour")["consumption"].mean().reset_index()
    fig_hourly = px.line(
        hourly_avg,
        x="hour",
        y="consumption",
        title="Avg Hourly Water Consumption",
        labels={"consumption": "Avg Consumption", "hour": "Hour"},
        markers=True,
        line_shape="spline"
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

with col2:
    st.markdown("### 📆 Weekly Pattern")
    weekday_avg = df.groupby("weekday")["consumption"].mean().reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]).reset_index()
    fig_week = px.bar(
        weekday_avg,
        x="weekday",
        y="consumption",
        title="Avg Consumption per Day of Week",
        labels={"consumption": "Avg Consumption", "weekday": "Day"}
    )
    st.plotly_chart(fig_week, use_container_width=True)

st.markdown("---")

<<<<<<< HEAD
# ---------- Detect Off-Peak High Usage ----------
=======
>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.subheader("🚨 High Off-Peak Usage Alerts")

off_peak_hours = list(range(0, 7))  # Midnight to 6 AM
off_peak_data = df[df["hour"].isin(off_peak_hours)]
threshold = df["consumption"].quantile(0.75)
high_off_peak = off_peak_data[off_peak_data["consumption"] > threshold]

if not high_off_peak.empty:
    st.warning(f"🚱 **{len(high_off_peak)} instances** of high usage during off-peak hours.")

    alert_summary = high_off_peak.groupby("hour")["consumption"].mean().reset_index()

    col3, col4 = st.columns([1, 1.5])

    with col3:
        st.markdown("#### 🧾 Table of Concern")
        st.dataframe(alert_summary.rename(columns={"hour": "Hour", "consumption": "Avg High Usage (Units)"}))

    with col4:
        fig_alert = px.bar(
            alert_summary,
            x="hour",
            y="consumption",
            color="consumption",
            color_continuous_scale="teal",
            title="High Usage During Off-Peak",
            labels={"hour": "Hour", "consumption": "Avg Usage"},
        )
        st.plotly_chart(fig_alert, use_container_width=True)

    st.markdown("### ✅ Recommendations")
    for _, row in alert_summary.iterrows():
        st.markdown(
            f"- 🔍 Check water systems around **{int(row['hour'])}:00** — avg usage `{row['consumption']:.1f}` units. Possible leak or inefficiency."
        )
else:
    st.success("🎉 No unusual off-peak water usage found. You're efficient!")

st.markdown("---")
#st.markdown("<br><br><hr><p style='text-align:left;'>Developed by Ismail Sadouki ❤️</p>", unsafe_allow_html=True)
