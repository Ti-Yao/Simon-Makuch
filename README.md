# Simon-Makuch Survival Analysis

This repository provides functions to generate a **Survival Event Table** and plot a **Simon-Makuch survival curve** for time-varying clusters. These functions allow for dynamic tracking of patient survival across different groups over time. The clusters can be easily adjusted to other time-varying covariates.

## Features
- **Creates a survival event table**: Counts patients at risk and deaths at each time point.
- **Plots Simon-Makuch survival curves**: Handles time-varying clusters.

## Installation
This script requires Python with the following libraries:

```bash
pip install numpy pandas matplotlib
```

## Usage
### 1. Create a Survival Event Table
```python
survival_table = create_survival_event_table(
    df, 'force_id', 'stop_death_transplant', 'status_death_transplant_varying', 'cluster'
)
```
#### Parameters:
- `df` : Pandas DataFrame containing patient data.
- `patient_col` : Column name for patient IDs.
- `stop_col` : Column for event time (death/transplant/censoring).
- `status_col` : Column for event status (1 = event, 0 = censored).
- `cluster_col` : Column indicating patient clusters.

#### Returns:
A **Pandas DataFrame** where each row represents a time point with the number of patients at risk and the number of deaths for each cluster.

---
### 2. Plot Simon-Makuch Survival Curves
```python
plot_simon_makuch(survival_table, ordered_colors=['blue', 'green', 'red'])
```
#### Parameters:
- `survival_table` : Output from `create_survival_event_table`.
- `ordered_colors` (optional) : List of colors for each cluster.


## Example Workflow
```python
import pandas as pd
from your_script import create_survival_event_table, plot_simon_makuch

# Load dataset
df = pd.read_csv('your_data.csv')

# Generate survival event table
survival_table = create_survival_event_table(df, 'force_id', 'stop_death_transplant', 'status_death_transplant_varying', 'cluster')

# Plot Simon-Makuch survival curves
plot_simon_makuch(survival_table)
```

## License
This project is open-source and free to use. Modify and distribute as needed.

## Contributions
Feel free to submit issues or pull requests to enhance the functionality!
