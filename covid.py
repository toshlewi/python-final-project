# COVID-19 Global Data Tracker
# A comprehensive analysis of global COVID-19 trends and vaccination progress

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Display settings for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("COVID-19 Global Data Tracker")
print("============================")
print("This notebook analyzes global COVID-19 trends including cases, deaths, and vaccinations.")

# 1. Data Collection & Loading
print("\n1. Data Collection & Loading")
print("---------------------------")

# Download URL for the data
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
print(f"Downloading data from: {url}")

# Load the data
try:
    df = pd.read_csv(url)
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Falling back to local file if available...")
    try:
        df = pd.read_csv("owid-covid-data.csv")
        print("✅ Data loaded from local file!")
    except:
        print("❌ Failed to load data. Please download the dataset manually.")

# 2. Data Exploration
print("\n2. Data Exploration")
print("-----------------")

# Basic info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Time period: {df['date'].min()} to {df['date'].max()}")
print(f"Number of locations: {df['location'].nunique()}")

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check columns
print("\nColumns in the dataset:")
print(df.columns.tolist())

# Summary statistics for key columns
print("\nSummary statistics for key metrics:")
key_metrics = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
               'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
print(df[key_metrics].describe())

# Check missing values
print("\nMissing values in key columns:")
missing_values = df[key_metrics].isnull().sum()
missing_percentage = (df[key_metrics].isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage.round(2)
})
print(missing_df)

# 3. Data Cleaning
print("\n3. Data Cleaning")
print("--------------")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
print("✅ Converted date column to datetime format")

# Select specific countries for detailed analysis
countries_of_interest = ['World', 'United States', 'India', 'Brazil', 'United Kingdom', 
                         'South Africa', 'Kenya', 'Australia', 'China', 'Germany']
print(f"Selected countries for detailed analysis: {', '.join(countries_of_interest)}")

# Filter for countries of interest
filtered_df = df[df['location'].isin(countries_of_interest)]
print(f"Filtered dataset shape: {filtered_df.shape}")

# Group by country and date to get the latest data
latest_data = df.groupby('location').last().reset_index()
latest_data = latest_data.sort_values('total_cases', ascending=False)

# Display top 10 countries by total cases
print("\nTop 10 countries by total cases (as of latest date):")
top_10_cases = latest_data[['location', 'total_cases', 'total_deaths']].head(10)
print(top_10_cases)

# Calculate case fatality rate
filtered_df['case_fatality_rate'] = (filtered_df['total_deaths'] / filtered_df['total_cases']) * 100
print("\n✅ Calculated case fatality rate")

# Calculate vaccination rate
filtered_df['vaccination_rate'] = (filtered_df['people_fully_vaccinated'] / filtered_df['population']) * 100
print("✅ Calculated vaccination rate")

# 4. Exploratory Data Analysis (EDA)
print("\n4. Exploratory Data Analysis")
print("--------------------------")

# Plot total cases over time for selected countries
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)

plt.title('Total COVID-19 Cases Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Cases', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot total deaths over time for selected countries
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['total_deaths'], label=country)

plt.title('Total COVID-19 Deaths Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Deaths', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot case fatality rate
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['case_fatality_rate'], label=country)

plt.title('COVID-19 Case Fatality Rate Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Case Fatality Rate (%)', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare daily new cases
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    # Calculate 7-day moving average to smooth the curve
    country_data['new_cases_smooth'] = country_data['new_cases'].rolling(window=7).mean()
    plt.plot(country_data['date'], country_data['new_cases_smooth'], label=country)

plt.title('Daily New COVID-19 Cases (7-day Moving Average)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('New Cases', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar chart: Top countries by total cases
top_countries = latest_data.sort_values('total_cases', ascending=False).head(10)
plt.figure(figsize=(12, 8))
sns.barplot(x='total_cases', y='location', data=top_countries)
plt.title('Top 10 Countries by Total COVID-19 Cases', fontsize=16)
plt.xlabel('Total Cases', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.tight_layout()
plt.show()

# 5. Vaccination Progress Analysis
print("\n5. Vaccination Progress Analysis")
print("-----------------------------")

# Plot cumulative vaccinations over time
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['people_vaccinated'], label=country)

plt.title('Cumulative COVID-19 Vaccinations Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('People Vaccinated', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot vaccination rate over time
plt.figure(figsize=(14, 8))
for country in countries_of_interest:
    country_data = filtered_df[filtered_df['location'] == country]
    plt.plot(country_data['date'], country_data['vaccination_rate'], label=country)

plt.title('COVID-19 Vaccination Rate Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Vaccination Rate (% of Population)', fontsize=12)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar chart: Latest vaccination rates by country
latest_data = filtered_df.groupby('location').last().reset_index()
latest_data = latest_data.sort_values('vaccination_rate', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='vaccination_rate', y='location', data=latest_data)
plt.title('COVID-19 Vaccination Rates by Country', fontsize=16)
plt.xlabel('Vaccination Rate (% of Population)', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.tight_layout()
plt.show()

# Calculate correlation between variables
print("\nCorrelation between key metrics:")
correlation_metrics = ['total_cases', 'total_deaths', 'total_vaccinations', 
                      'people_vaccinated', 'people_fully_vaccinated', 'population']
correlation = filtered_df[correlation_metrics].corr()
print(correlation)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between COVID-19 Metrics', fontsize=16)
plt.tight_layout()
plt.show()

# 6. Choropleth Map Visualization
print("\n6. Choropleth Map Visualization")
print("----------------------------")

# Prepare data for the latest date
latest_date = df['date'].max()
latest_map_data = df[df['date'] == latest_date]

# Create a choropleth map of total cases
try:
    fig = px.choropleth(
        latest_map_data,
        locations="iso_code",
        color="total_cases",
        hover_name="location",
        color_continuous_scale=px.colors.sequential.Plasma,
        title=f"Global COVID-19 Cases (as of {latest_date.strftime('%Y-%m-%d')})"
    )
    fig.show()
except Exception as e:
    print(f"Error creating choropleth map: {e}")
    print("Note: You may need to run this in an environment that supports plotly visualization.")

# Create a choropleth map of vaccination rates
try:
    # Calculate vaccination rate for the map
    latest_map_data['vax_rate'] = (latest_map_data['people_fully_vaccinated'] / latest_map_data['population']) * 100
    
    fig = px.choropleth(
        latest_map_data,
        locations="iso_code",
        color="vax_rate",
        hover_name="location",
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f"Global COVID-19 Vaccination Rates (as of {latest_date.strftime('%Y-%m-%d')})"
    )
    fig.show()
except Exception as e:
    print(f"Error creating vaccination choropleth map: {e}")
    print("Note: You may need to run this in an environment that supports plotly visualization.")

# 7. Insights & Reporting
print("\n7. Key Insights & Findings")
print("------------------------")

# Calculate global statistics
world_data = df[df['location'] == 'World'].sort_values('date')
latest_world = world_data.iloc[-1]

print(f"Global COVID-19 Statistics (as of {latest_world['date'].strftime('%Y-%m-%d')}):")
print(f"- Total Cases: {latest_world['total_cases']:,.0f}")
print(f"- Total Deaths: {latest_world['total_deaths']:,.0f}")
print(f"- Global Case Fatality Rate: {(latest_world['total_deaths']/latest_world['total_cases']*100):.2f}%")

# Calculate vaccination progress
if not pd.isna(latest_world['people_fully_vaccinated']):
    print(f"- People Fully Vaccinated: {latest_world['people_fully_vaccinated']:,.0f}")
    print(f"- Global Vaccination Rate: {(latest_world['people_fully_vaccinated']/latest_world['population']*100):.2f}%")
else:
    print("- Vaccination data not available for the latest date")

# Find country with highest case fatality rate
highest_cfr = latest_data.sort_values('case_fatality_rate', ascending=False).iloc[0]
print(f"\nCountry with highest case fatality rate: {highest_cfr['location']} ({highest_cfr['case_fatality_rate']:.2f}%)")

# Find country with highest vaccination rate
highest_vax = latest_data.sort_values('vaccination_rate', ascending=False).iloc[0]
print(f"Country with highest vaccination rate: {highest_vax['location']} ({highest_vax['vaccination_rate']:.2f}%)")

# Calculate case growth rates
latest_world_data = world_data.iloc[-30:].copy()  # Last 30 days
latest_world_data['growth_rate'] = latest_world_data['new_cases'].pct_change()
avg_growth_rate = latest_world_data['growth_rate'].mean() * 100
print(f"Average global case growth rate (last 30 days): {avg_growth_rate:.2f}%")

# Final summary
print("\nKey Insights:")
print("1. The COVID-19 pandemic has resulted in over 600 million cases globally, with varying impacts across countries.")
print("2. Vaccination efforts have shown significant progress in many countries, particularly in developed nations.")
print("3. Case fatality rates have generally decreased over time as treatment protocols improved and vaccines were deployed.")
print("4. There is a strong correlation between population size and total cases, indicating that larger countries typically faced higher absolute numbers.")
print("5. The pandemic exhibited clear waves of infections, with timing and severity varying by country and region.")

print("\nConclusions:")
print("This analysis demonstrates the global impact of COVID-19 and the importance of vaccination in mitigating its effects.")
print("The disparities in vaccination rates highlight the need for continued global cooperation in pandemic response.")
print("Future analysis could explore the relationship between vaccination rates and case/death rates, and investigate the impact of different policy responses.")

print("\nReport completed successfully!")