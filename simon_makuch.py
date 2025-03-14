import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_truncate_index(lst):
    if not lst:
        return None  # Handle empty list case

    last_value = lst[-1]  # The repeating value at the end
    for i in range(len(lst) - 2, -1, -1):  # Traverse backward
        if lst[i] != last_value:
            return i + 1  # The first index of the repeating sequence
    return 0  # If the entire list is the same value

def create_survival_event_table(df, patient_col, stop_col, status_col, cluster_col):
    """
    Generates a survival event table tracking patient counts and deaths over time.

    Parameters:
        df (DataFrame): The input dataset.
        patient_col (str): Column name for patient IDs.
        stop_col (str): Column name for event time (death/transplant/censoring).
        status_col (str): Column name for event status (1 = event, 0 = censored).
        cluster_col (str): Column name for cluster/grouping.

    Returns:
        DataFrame: A survival event table with patient and death counts over time.
    """
    # Filter required columns and drop NaN values
    event_df = df[[patient_col, stop_col, status_col, cluster_col]].dropna()

    # Sort by event time
    event_df = event_df.sort_values(stop_col)

    # Extract unique time points
    time_points = np.sort(event_df[stop_col].unique())

    # Get unique clusters
    clusters = np.arange(event_df[cluster_col].nunique())

    # Initialize results list
    results = []

    # Initial patient count per cluster
    all_patients = {cluster: event_df[event_df[cluster_col] == cluster][patient_col].nunique() for cluster in clusters}

    # Create initial row (time = 0)
    row = {'time': 0}
    for cluster in all_patients.keys():
        row[f'cluster_{cluster}_n_patients'] = all_patients[cluster]
        row[f'cluster_{cluster}_n_deaths'] = 0  # No deaths at time 0
    results.append(row)

    # Iterate over each event time point
    for time in time_points:
        # Patients alive at this time
        alive_at_time = event_df[event_df[stop_col] >= time]
        cluster_counts = alive_at_time.groupby(cluster_col)[patient_col].nunique().to_dict()

        # Count deaths at this time
        deaths_at_time = event_df[(event_df[stop_col] == time) & (event_df[status_col] == 1)]
        cluster_deaths = deaths_at_time.groupby(cluster_col)[patient_col].nunique().to_dict()

        # Update patient counts
        for cluster in cluster_counts.keys():
            all_patients[cluster] = cluster_counts.get(cluster, 0)

        # Create row for this time
        row = {'time': time}
        for cluster in all_patients.keys():
            row[f'cluster_{cluster}_n_patients'] = all_patients[cluster]
            row[f'cluster_{cluster}_n_deaths'] = cluster_deaths.get(cluster, 0)
        results.append(row)

    # Convert results to DataFrame
    final_table = pd.DataFrame(results)

    # Set time as index
    final_table.set_index('time', inplace=True)

    # Reshape to multi-index columns
    final_table.columns = pd.MultiIndex.from_tuples(
        [(f'Cluster {cluster + 1}', metric) for cluster in clusters for metric in ['n_patients', 'n_deaths']],
        names=['cluster', 'metric']
    )

    return final_table


def plot_simon_makuch(survival_table, ordered_colors = None):
    """
    Plots the Simon-Makuch survival curves based on the provided survival event table.

    Parameters:
        survival_table (DataFrame): Survival event table with patient and death counts.
        ordered_colors (list): List of colors for each cluster.
    """
    plt.figure(figsize=(8, 6))
    labels = []
    min_survival = 1.0  # Track the minimum survival probability


    for cluster_id, cluster in enumerate(survival_table.columns.levels[0]):
        # Initialize survival probability
        survival_probabilities = [1.0]
        patients_at_risk = [survival_table[(cluster, 'n_patients')].iloc[0]]
        death_counts = [survival_table[(cluster, 'n_deaths')].iloc[0]]

        # Calculate survival probabilities
        for i in range(1, len(survival_table)):
            patients_at_risk.append(survival_table[(cluster, 'n_patients')].iloc[i])
            death_counts.append(survival_table[(cluster, 'n_deaths')].iloc[i])

            deaths_today = death_counts[i]
            at_risk_today = patients_at_risk[i - 1]
            survival_today = survival_probabilities[i - 1] * (1 - deaths_today / at_risk_today)
            survival_probabilities.append(survival_today)

        # Optionally truncate at meaningful index
        truncate_index = find_truncate_index(survival_probabilities)
        survival_probabilities = survival_probabilities[:truncate_index]
        times = survival_table.index[:truncate_index]

        # Plot survival function
        labels.append(cluster)
        if ordered_colors:
            plt.step(times, survival_probabilities, where='post', color=ordered_colors[cluster_id], linewidth=4)
        else:
            plt.step(times, survival_probabilities, where='post', linewidth=4)

        # Track the minimum survival probability
        min_survival = min(min_survival, min(survival_probabilities))

    # Set y-axis ticks starting from np.floor(min_survival) with 0.1 intervals
    y_min = np.floor(min_survival * 10) / 10  # Floor to nearest 0.1
    plt.yticks(np.arange(y_min, 1.1, 0.1))  # Ticks from y_min to 1 in 0.1 steps
    plt.ylim(y_min,1)
    
    # Customize plot
    plt.title("Simon-Makuch Plot", fontsize = 20)
    plt.xlabel("Time (Days)", fontsize=16)
    plt.ylabel("Survival Probability", fontsize=16)
    plt.legend(loc='lower left', labels=labels, framealpha=0, fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()
