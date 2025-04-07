from imblearn.under_sampling import NearMiss, RepeatedEditedNearestNeighbours
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data and check some general information
data_lending_ca = pd.read_csv(r"C:\Users\koles\Desktop\Master Thesis R project\hmda_2017_ca_all-records_labels.csv", low_memory=False)
print(data_lending_ca.shape)
print(data_lending_ca.columns)
print(data_lending_ca['action_taken_name'].value_counts())

# Values to remove
values_to_remove = {
    'action_taken': [2, 4, 5, 6, 7, 8],
    'loan_type': [3, 4],
    'applicant_race_1': [1, 4, 6, 7],
    'lien_status': [3, 4],
    'applicant_sex': [3, 4]
}

# Filter
data_lending_ca_short = data_lending_ca.copy()
for col, values in values_to_remove.items():
    data_lending_ca_short = data_lending_ca_short[~data_lending_ca_short[col].isin(values)]
print(data_lending_ca_short.shape)
print(data_lending_ca_short['action_taken_name'].value_counts())

# Delete and refactor columns
columns_to_remove = ['as_of_year', 'agency_name', 'agency_abbr', 'agency_code', 'property_type_name',
       'property_type', 'owner_occupancy_name', 'owner_occupancy', 'preapproval_name', 'preapproval',
       'state_name', 'state_abbr', 'state_code', 'applicant_race_name_2', 'applicant_race_2',
       'applicant_race_name_3', 'applicant_race_3', 'applicant_race_name_4', 'applicant_race_4',
       'applicant_race_name_5', 'applicant_race_5', 'co_applicant_race_name_2', 'co_applicant_race_2',
       'co_applicant_race_name_3', 'co_applicant_race_3', 'co_applicant_race_name_4', 'co_applicant_race_4',
       'co_applicant_race_name_5', 'co_applicant_race_5', 'purchaser_type_name', 'purchaser_type',
       'denial_reason_name_1', 'denial_reason_1', 'denial_reason_name_2', 'denial_reason_2', 'denial_reason_name_3',
       'denial_reason_3', 'rate_spread', 'hoepa_status_name', 'hoepa_status','edit_status_name', 'edit_status',
       'sequence_number', 'application_date_indicator', 'respondent_id', 'loan_type_name', 'loan_purpose_name',
       'action_taken_name', 'msamd_name', 'county_name', 'applicant_ethnicity_name', 'co_applicant_ethnicity_name',
       'applicant_race_name_1', 'co_applicant_race_name_1', 'applicant_sex_name', 'co_applicant_sex_name',
       'lien_status_name']

data_lending_ca_dropped = data_lending_ca_short.drop(columns=columns_to_remove)
print(data_lending_ca_dropped.shape)
print(data_lending_ca_dropped.columns)

types = data_lending_ca_dropped.dtypes
print(types)
print(data_lending_ca_dropped.isna().sum())  # Number of NaNs per column
print(data_lending_ca_dropped.isna().sum().sum())  # Total number of NaNs in the entire data

# Clean data
data_lending_ca_clean = data_lending_ca_dropped.dropna(axis='index')
data_lending_ca_clean.loc[:, 'action_taken'] = data_lending_ca_clean['action_taken'].map({1: 0, 3: 1})
print(data_lending_ca_clean.shape)
print(data_lending_ca_clean['action_taken'].value_counts())

# Check correlations
correlation_matrix = data_lending_ca_clean.corr()
print(correlation_matrix['action_taken'])

# Data separation
x = data_lending_ca_clean.drop(columns=['action_taken'])
y = data_lending_ca_clean['action_taken']

# Apply Edited Nearest Neighbors undersampling
renn = RepeatedEditedNearestNeighbours(n_neighbors=4, n_jobs=-1)
x_resampled, y_resampled = renn.fit_resample(x, y)

# Apply NearMiss and get 50/50 class distribution
nm = NearMiss(sampling_strategy=1, version=1, n_jobs=-1)
x_balanced, y_balanced = nm.fit_resample(x_resampled, y_resampled)

# Convert back to DataFrame and save
data_lending_ca_resampled = pd.DataFrame(x_balanced, columns=x.columns)
data_lending_ca_resampled['action_taken'] = y_balanced.reset_index(drop=True)
data_lending_ca_resampled['action_taken'] = data_lending_ca_resampled['action_taken'].astype(int)
data_lending_ca_resampled.to_csv(os.path.join("../saved_data", "data_lending_resampled_ca.csv"), index=False)

# Print class distribution
print("Class distribution after ENN and NearMiss:")
print(data_lending_ca_resampled['action_taken'].value_counts())

# Standardize the features before PCA
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_resampled_scaled = scaler.transform(x_balanced)

# Apply PCA
pca = PCA(n_components=2)
x_pca_before = pca.fit_transform(x_scaled)
x_pca_after = pca.transform(x_resampled_scaled)

# Plot PCA
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(x_pca_before[:, 0], x_pca_before[:, 1], c=y, cmap='coolwarm', s=30, alpha=0.7)
axes[0].set_title("Before Undersampling")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

axes[1].scatter(x_pca_after[:, 0], x_pca_after[:, 1], c=y_balanced, cmap='coolwarm', s=30, alpha=0.7)
axes[1].set_title("After Undersampling")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

# Create folder for saved data
save_dir = "../saved_data"
os.makedirs(save_dir, exist_ok=True)