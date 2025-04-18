import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
from sklearn.metrics import silhouette_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Streamlit setup
st.set_page_config(page_title="🌿 Smart Sustainability Dashboard", layout="wide")
st.title("🌱 Smart Sustainability Optimization")

# Load Data
@st.cache_data
def load_all():
    # Read data files, parse dates correctly, and add additional time-related columns
    bld = pd.read_csv("building_consumption.csv", parse_dates=["timestamp"])
    gas = pd.read_csv("gas_consumption.csv", parse_dates=["timestamp"])
    water = pd.read_csv("water_consumption.csv", parse_dates=["timestamp"])

    # Generate daily and hourly columns
    for df in [bld, gas, water]:
        df["day"] = df["timestamp"].dt.floor("D")
        df["hour"] = df["timestamp"].dt.hour

    return bld, gas, water

building, gas, water = load_all()

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
campus = st.sidebar.selectbox("Select Campus", sorted(building['campus_id'].unique()))

# Customizable Policy Effects
st.sidebar.subheader("Customize Policy Impact")
heating_reduction = st.sidebar.slider("Heating Reduction (%)", 0, 20, 10, 1) / 100
water_efficiency = st.sidebar.slider("Water Efficiency (%)", 0, 50, 15, 1) / 100
solar_efficiency = st.sidebar.slider("Solar Panel Efficiency (%)", 0, 50, 25, 1) / 100
hours_reduction = st.sidebar.slider("Building Hours Reduction (%)", 0, 50, 10, 1) / 100

# Customizable Simulation Parameters
st.sidebar.subheader("Simulation Parameters")
years = st.sidebar.slider("Number of Years", 1, 50, 10, 1)
simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, 100)

# Variable Randomization
st.sidebar.subheader("Consumption Variability")
electricity_variability = st.sidebar.slider("Electricity Consumption Variability (%)", 0, 20, 5, 1) / 100
gas_variability = st.sidebar.slider("Gas Consumption Variability (%)", 0, 20, 5, 1) / 100
water_variability = st.sidebar.slider("Water Consumption Variability (%)", 0, 20, 5, 1) / 100

# CO₂ factors (could be adjusted by the user)
st.sidebar.subheader("CO₂ Emission Factors (kg per unit)")
electricity_co2_factor = st.sidebar.number_input("Electricity CO₂ Factor (kg/kWh)", value=0.233)
gas_co2_factor = st.sidebar.number_input("Gas CO₂ Factor (kg/m³)", value=2.204)
water_co2_factor = st.sidebar.number_input("Water CO₂ Factor (kg/L)", value=0.0015)

# Policies selected by user
policy = st.sidebar.multiselect("Apply Policies", 
    ["Reduce Heating by 10%", "Efficient Water Fixtures", "Solar Panels Installed", "Shorten Building Hours"])

# Filter data based on selected campus
bld_c = building[building["campus_id"] == campus]
gas_c = gas[gas["campus_id"] == campus]
water_c = water[water["campus_id"] == campus]

# Benchmarking: Compare current campus with others
st.subheader("📊 Building Consumption Benchmarking")
grouped = building.groupby("campus_id")["consumption"].mean()
best = grouped.nsmallest(1).values[0]
campus_avg = grouped[campus]

efficiency_ratio = best / campus_avg
st.metric("Efficiency Score (1=Best)", f"{efficiency_ratio:.2f}")

fig = px.bar(grouped, title="Average Electricity Consumption by Campus")
st.plotly_chart(fig, use_container_width=True)

# CO₂ Emissions Estimation: Model based on consumption
st.subheader("🌍 CO₂ Emissions Estimation")
total_co2 = (
    bld_c["consumption"].sum() * electricity_co2_factor
    + gas_c["consumption"].sum() * gas_co2_factor
    + water_c["consumption"].sum() * water_co2_factor
)

# Impact of applied policies
policy_impact = 0
impact_details = {}

if "Reduce Heating by 10%" in policy:
    heating_impact = heating_reduction * gas_c["consumption"].sum() * gas_co2_factor
    policy_impact += heating_impact
    impact_details["Heating Reduction"] = heating_impact

if "Efficient Water Fixtures" in policy:
    water_impact = water_efficiency * water_c["consumption"].sum() * water_co2_factor
    policy_impact += water_impact
    impact_details["Water Efficiency"] = water_impact

if "Solar Panels Installed" in policy:
    solar_impact = solar_efficiency * bld_c["consumption"].sum() * electricity_co2_factor
    policy_impact += solar_impact
    impact_details["Solar Panel Efficiency"] = solar_impact

if "Shorten Building Hours" in policy:
    hours_impact = hours_reduction * bld_c["consumption"].sum() * electricity_co2_factor
    policy_impact += hours_impact
    impact_details["Building Hours Reduction"] = hours_impact

# CO₂ savings due to policies
reduction = total_co2 - policy_impact
st.metric("Estimated CO₂ After Policies (kg)", f"{reduction:,.2f}")
st.metric("CO₂ Saved Due to Policies (kg)", f"{policy_impact:,.2f}")

# Show Real CO₂ Emissions Before and After Policies
st.subheader("🌿 Real and Simulated CO₂ Emissions Comparison")

# Table with real, impact, and benefit details
real_data = {
    "Initial CO₂ Emissions (kg)": total_co2,
    "CO₂ Saved by Policies (kg)": policy_impact,
    "Final CO₂ Emissions After Policies (kg)": reduction
}
st.write(pd.DataFrame([real_data]))

# Visualize the Impact of Policies on CO₂ Emissions
st.subheader("📊 Policy Impact Visualization")

# Create a bar chart for visualizing the policies' effects
policy_names = list(impact_details.keys())
impact_values = list(impact_details.values())

impact_df = pd.DataFrame({"Policy": policy_names, "Impact (kg CO2)": impact_values})
impact_df = impact_df[impact_df["Impact (kg CO2)"] > 0]  # Filter out non-applied policies

fig = px.bar(impact_df, x="Policy", y="Impact (kg CO2)", title="Impact of Policies on CO₂ Emissions", color="Policy")
st.plotly_chart(fig, use_container_width=True)

# Policy Simulation: Monte Carlo Simulation to model policy impact over customizable years
st.subheader("💡 Policy Simulation (Monte Carlo)")

# Define parameters for the Monte Carlo simulation
base_consumption = {
    "electricity": bld_c["consumption"].mean(),
    "gas": gas_c["consumption"].mean(),
    "water": water_c["consumption"].mean()
}

# Simulation function
def run_simulation(base_consumption, policy_impact, years=10, simulations=1000):
    np.random.seed(42)
    simulated_results = []
    
    for _ in range(simulations):
        yearly_changes = []
        
        for year in range(years):
            # Apply random percentage change (simulate real-world variability)
            electricity_change = np.random.normal(0, electricity_variability)  # Variability in electricity consumption
            gas_change = np.random.normal(0, gas_variability)  # Variability in gas consumption
            water_change = np.random.normal(0, water_variability)  # Variability in water consumption
            
            # Adjust consumption based on policies
            simulated_electricity = base_consumption["electricity"] * (1 + electricity_change)
            simulated_gas = base_consumption["gas"] * (1 + gas_change)
            simulated_water = base_consumption["water"] * (1 + water_change)
            
            # Apply policy impacts (e.g., reducing heating or adding solar panels)
            if "Reduce Heating by 10%" in policy:
                simulated_gas *= (1 - heating_reduction)  # Apply custom heating reduction
            if "Efficient Water Fixtures" in policy:
                simulated_water *= (1 - water_efficiency)  # Apply custom water efficiency
            if "Solar Panels Installed" in policy:
                simulated_electricity *= (1 - solar_efficiency)  # Apply custom solar efficiency
            if "Shorten Building Hours" in policy:
                simulated_electricity *= (1 - hours_reduction)  # Apply custom building hours reduction
            
            # Calculate CO₂ emissions for this year
            simulated_co2 = (
                simulated_electricity * electricity_co2_factor +
                simulated_gas * gas_co2_factor +
                simulated_water * water_co2_factor
            )
            yearly_changes.append(simulated_co2)
        
        # Store the results of the simulation
        simulated_results.append(np.sum(yearly_changes))
    
    return simulated_results

# Run the simulation
simulation_results = run_simulation(base_consumption, policy_impact, years, simulations)

# Display simulation results
st.write(f"Simulated Total CO₂ Emissions Over {years} Years ({simulations} Simulations)")

# Plot the simulation results (distribution of total CO₂ over the 10-year period)
fig = px.histogram(simulation_results, nbins=30, title="Simulated CO₂ Emissions Over 10 Years")
st.plotly_chart(fig, use_container_width=True)

# Display average result
avg_emissions = np.mean(simulation_results)
st.metric("Average Total CO₂ Emissions (kg)", f"{avg_emissions:,.2f}")

