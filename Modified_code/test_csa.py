import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Added for enhanced statistical visualization
from csa import csa

import os

# Create results directory if it does not exist
os.makedirs('results', exist_ok=True)

# Read MIP, POPS, and CDP CSVs in test_data folder
mip_data_df = pd.read_csv(r'test_data\mip_data.csv')
pops_data_df = pd.read_csv(r'test_data\pops_data.csv')
cdp_data_df = pd.read_csv(r'test_data\cdp_data.csv')

# Initialize lists to store results for plotting
results = {
    'time': [],
    'latitude': [],
    'longitude': [],
    'cdp_number_conc': [],
    'cdp_lwc': [],
    'cdp_mvd': [],
    'seed_score': [],
    'seed_switch': [],
    'vertical_wind': [],
    'pops_number_conc': []
}

# Process each row of data
for count in range(len(mip_data_df)):
    mip_data = mip_data_df.iloc[count, :].values.tolist()
    pops_data = pops_data_df.iloc[count, :].values.tolist()
    cdp_data = cdp_data_df.iloc[count, :].values.tolist()

    # Call CSA function
    csa_out = csa(mip_data, pops_data, cdp_data)

    # Store results
    results['time'].append(csa_out[0])
    results['latitude'].append(csa_out[1])
    results['longitude'].append(csa_out[2])
    results['cdp_number_conc'].append(csa_out[3])
    results['cdp_lwc'].append(csa_out[4])
    results['cdp_mvd'].append(csa_out[5])
    results['seed_score'].append(csa_out[6])
    results['seed_switch'].append(csa_out[7])
    results['vertical_wind'].append(mip_data[39])  # Assuming mip_data[39] is vertical wind
    results['pops_number_conc'].append(pops_data[3])  # Assuming pops_data[3] is POPS number concentration

    # Print results for each iteration
    print(csa_out)

# Convert results to DataFrame for further analysis
results_df = pd.DataFrame(results)

# Save results to a CSV file for research purposes
results_df.to_csv('results/csa_results.csv', index=False)

# Save all figures in the results folder
output_folder = 'results/'

# Plotting for Research Paper

# Seed Score Time Series
plt.figure(figsize=(10, 6))
plt.plot(results['time'], results['seed_score'], label='Seed Score', color='b')
plt.axhline(y=8, color='r', linestyle='--', label='Seed Score Threshold')
plt.xlabel('Time (seconds in day)')
plt.ylabel('Seed Score')
plt.title('Seed Score Time Series')
plt.legend()
plt.grid(True)
plt.savefig(output_folder + 'seed_score_time_series.png')  # Save figure
plt.show()

# Seed Score Distribution
plt.figure(figsize=(10, 6))
plt.hist(results['seed_score'], bins=20, color='g', alpha=0.7)
plt.axvline(x=8, color='r', linestyle='--', label='Seed Score Threshold')
plt.xlabel('Seed Score')
plt.ylabel('Frequency')
plt.title('Distribution of Seed Scores')
plt.legend()
plt.grid(True)
plt.savefig(output_folder + 'seed_score_distribution.png')  # Save figure
plt.show()

# CDP Number Concentration Time Series
plt.figure(figsize=(10, 6))
plt.plot(results['time'], results['cdp_number_conc'], label='CDP Number Concentration', color='m')
plt.xlabel('Time (seconds in day)')
plt.ylabel('CDP Number Concentration (cm^-3)')
plt.title('CDP Number Concentration Time Series')
plt.legend()
plt.grid(True)
plt.savefig(output_folder + 'cdp_number_concentration_time_series.png')  # Save figure
plt.show()

# Vertical Wind Time Series
plt.figure(figsize=(10, 6))
plt.plot(results['time'], results['vertical_wind'], label='Vertical Wind', color='c')
plt.xlabel('Time (seconds in day)')
plt.ylabel('Vertical Wind (m/s)')
plt.title('Vertical Wind Time Series')
plt.legend()
plt.grid(True)
plt.savefig(output_folder + 'vertical_wind_time_series.png')  # Save figure
plt.show()

# POPS Number Concentration Time Series
plt.figure(figsize=(10, 6))
plt.plot(results['time'], results['pops_number_conc'], label='POPS Number Concentration', color='y')
plt.xlabel('Time (seconds in day)')
plt.ylabel('POPS Number Concentration (cm^-3)')
plt.title('POPS Number Concentration Time Series')
plt.legend()
plt.grid(True)
plt.savefig(output_folder + 'pops_number_concentration_time_series.png')  # Save figure
plt.show()

# Additional Analysis Plots

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = results_df[['cdp_number_conc', 'cdp_lwc', 'cdp_mvd', 'seed_score', 'vertical_wind', 'pops_number_conc']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Key Parameters')
plt.savefig(output_folder + 'correlation_heatmap.png')  # Save figure
plt.show()

# Box Plots for Distribution Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(data=results_df[['cdp_number_conc', 'vertical_wind', 'pops_number_conc', 'seed_score']])
plt.title('Box Plots of Key Parameters')
plt.ylabel('Values')
plt.grid(True)
plt.savefig(output_folder + 'box_plots.png')  # Save figure
plt.show()

# Rolling Mean and Standard Deviation for Seed Score
results_df['rolling_mean_seed_score'] = results_df['seed_score'].rolling(window=10).mean()
results_df['rolling_std_seed_score'] = results_df['seed_score'].rolling(window=10).std()

plt.figure(figsize=(10, 6))
plt.plot(results['time'], results_df['rolling_mean_seed_score'], label='Rolling Mean Seed Score', color='b')
plt.fill_between(results['time'], results_df['rolling_mean_seed_score'] - results_df['rolling_std_seed_score'],
                 results_df['rolling_mean_seed_score'] + results_df['rolling_std_seed_score'], color='b', alpha=0.2)
plt.xlabel('Time (seconds in day)')
plt.ylabel('Seed Score')
plt.title('Rolling Mean and Standard Deviation of Seed Score')
plt.legend()
plt.grid(True)
plt.savefig(output_folder + 'rolling_mean_std_seed_score.png')  # Save figure
plt.show()

# Seed Score vs CDP Number Concentration
plt.figure(figsize=(10, 6))
plt.scatter(results['cdp_number_conc'], results['seed_score'], alpha=0.5)
plt.xlabel('CDP Number Concentration (cm^-3)')
plt.ylabel('Seed Score')
plt.title('Seed Score vs. CDP Number Concentration')
plt.grid(True)
plt.savefig(output_folder + 'seed_score_vs_cdp_number_concentration.png')  # Save figure
plt.show()

# Seed Score vs Vertical Wind
plt.figure(figsize=(10, 6))
plt.scatter(results['vertical_wind'], results['seed_score'], alpha=0.5)
plt.xlabel('Vertical Wind (m/s)')
plt.ylabel('Seed Score')
plt.title('Seed Score vs. Vertical Wind')
plt.grid(True)
plt.savefig(output_folder + 'seed_score_vs_vertical_wind.png')  # Save figure
plt.show()

# Seed Score vs POPS Number Concentration
plt.figure(figsize=(10, 6))
plt.scatter(results['pops_number_conc'], results['seed_score'], alpha=0.5)
plt.xlabel('POPS Number Concentration (cm^-3)')
plt.ylabel('Seed Score')
plt.title('Seed Score vs. POPS Number Concentration')
plt.grid(True)
plt.savefig(output_folder + 'seed_score_vs_pops_number_concentration.png')  # Save figure
plt.show()

# Smoothed Time Series for Key Parameters
plt.figure(figsize=(10, 6))
sns.lineplot(x='time', y='cdp_number_conc', data=results_df, label='CDP Number Conc. (Smoothed)', color='m', ci=None)
sns.lineplot(x='time', y='vertical_wind', data=results_df, label='Vertical Wind (Smoothed)', color='c', ci=None)
sns.lineplot(x='time', y='pops_number_conc', data=results_df, label='POPS Number Conc. (Smoothed)', color='y', ci=None)
plt.xlabel('Time (seconds in day)')
plt.ylabel('Values')
plt.title('Smoothed Time Series of Key Parameters')
plt.legend()
plt.grid(True)
plt.savefig(output_folder + 'smoothed_time_series.png')  # Save figure
plt.show()
