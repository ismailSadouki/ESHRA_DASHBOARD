import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
<<<<<<< HEAD

# App configuration
=======
from utils import DataManager




>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.set_page_config(
    page_title="AI-Powered Sustainability Dashboards",
    page_icon="⚡",  # 🏢 = building emoji
    layout="wide",
)

<<<<<<< HEAD
st.title("🏢 AI-Powered Sustainability Dashboard")

# Welcome Section with Icons
=======
# Instantiate the DataManager
dm = DataManager()
# Load data (only once)
#energy_data = dm.load_energy()


st.title("🏢 AI-Powered Sustainability Dashboard")

>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.markdown("""
    ### Optimize your building's resources with data-driven insights
    Unlock **cost-saving** and **energy-efficient** solutions for:
    - 🛢️ **Gas Consumption**
    - 💧 **Water Usage**
    - ⚡ **Energy Efficiency**
""")

<<<<<<< HEAD
# Interactive Elements
=======
>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.sidebar.header("🔧 Select Optimization Focus")
optimization_method = st.sidebar.selectbox(
    "Choose an area to optimize:",
    ["Gas Consumption", "Water Usage", "Energy Efficiency"]
)

<<<<<<< HEAD
# Dummy Data for Graphs
# Here we simulate some data for consumption trends over time
=======
>>>>>>> 4cd54e7 (Initial commit after reinitializing)
time = pd.date_range("2024-01-01", periods=100, freq="H")
gas_consumption = np.random.uniform(20, 50, size=100)
water_consumption = np.random.uniform(10, 30, size=100)
energy_consumption = np.random.uniform(50, 100, size=100)

df = pd.DataFrame({"timestamp": time, "gas": gas_consumption, "water": water_consumption, "energy": energy_consumption})

<<<<<<< HEAD
# Graphs Section Based on User Selection
=======
>>>>>>> 4cd54e7 (Initial commit after reinitializing)
if optimization_method == "Gas Consumption":
    st.subheader("🛢️ Gas Consumption Trend")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['gas'], mode='lines', name='Gas Consumption', line=dict(color='blue')))
    fig1.update_layout(title="Gas Consumption Over Time", xaxis_title="Timestamp", yaxis_title="Consumption (m³)")
    st.plotly_chart(fig1, use_container_width=True)

elif optimization_method == "Water Usage":
    st.subheader("💧 Water Consumption Trend")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['water'], mode='lines', name='Water Consumption', line=dict(color='green')))
    fig2.update_layout(title="Water Consumption Over Time", xaxis_title="Timestamp", yaxis_title="Consumption (m³)")
    st.plotly_chart(fig2, use_container_width=True)

elif optimization_method == "Energy Efficiency":
    st.subheader("⚡ Energy Consumption Trend")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['timestamp'], y=df['energy'], mode='lines', name='Energy Consumption', line=dict(color='red')))
    fig3.update_layout(title="Energy Consumption Over Time", xaxis_title="Timestamp", yaxis_title="Consumption (kWh)")
    st.plotly_chart(fig3, use_container_width=True)

<<<<<<< HEAD
# Simple Call to Action
=======
>>>>>>> 4cd54e7 (Initial commit after reinitializing)
st.markdown("""
    ### Take control of your building's energy footprint today
    Choose a focus area from the sidebar and explore the **data-driven solutions** that can drive **significant savings** and **efficiency improvements**.
""")

<<<<<<< HEAD
# Footer with Contact or Info Link
#st.markdown("<br><br><hr><p style='text-align:left;'>Developed by Ismail Sadouki ❤️</p>", unsafe_allow_html=True)

=======
#st.markdown("<br><br><hr><p style='text-align:left;'>Developed by Ismail Sadouki ❤️</p>", unsafe_allow_html=True)






>>>>>>> 4cd54e7 (Initial commit after reinitializing)
