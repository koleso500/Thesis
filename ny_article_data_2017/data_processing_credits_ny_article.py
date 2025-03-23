import os
import pandas as pd

# Load data and check some general information
data_lending_ny = pd.read_csv(r"C:\Users\koles\Desktop\Master Thesis\data_lending.csv", low_memory=False)
print(data_lending_ny.shape)
print(data_lending_ny.columns)
print(data_lending_ny['response'].value_counts())

# Delete and refactor columns
columns_to_remove = ['Unnamed: 0', 'X', 'respondent_id', 'action_taken_name', 'applicant_race_2',
             'applicant_race_3', 'applicant_race_4', 'applicant_race_name_5',
             'applicant_race_5', 'co_applicant_race_2', 'co_applicant_race_3', 'co_applicant_race_name_4',
             'co_applicant_race_4', 'co_applicant_race_name_5', 'co_applicant_race_5', 'denial_reason_name_1',
             'denial_reason_1', 'denial_reason_name_2', 'denial_reason_2', 'denial_reason_name_3',
             'denial_reason_3', 'rate_spread', 'applicant_race_name_2', 'applicant_race_name_3',
             'applicant_race_name_4', 'co_applicant_race_name_2', 'co_applicant_race_name_3','loan_type_name',
             'loan_purpose_name', 'action_taken', 'msamd_name','county_name','applicant_ethnicity_name',
             'co_applicant_ethnicity_name','applicant_race_name_1', 'co_applicant_race_name_1', 'lien_status_name',
             'purchaser_type_name', 'purchaser_type']

data_lending_ny_short = data_lending_ny.drop(columns=columns_to_remove)
data_lending_ny_short['applicant_sex_name'] = data_lending_ny_short['applicant_sex_name'].map({'Male': 1, 'Female': 0})
data_lending_ny_short['co_applicant_sex_name'] = data_lending_ny_short['co_applicant_sex_name'].map({'Female': 0, 'Male': 1, 'No co-applicant': 2,
       'Information not provided by applicant in mail, Internet, or telephone application': 3,
       'Not applicable': 4})
print(data_lending_ny_short.columns)

types = data_lending_ny_short.dtypes
print(types)
print(data_lending_ny_short.isna().sum())  # Number of NaNs per column
print(data_lending_ny_short.isna().sum().sum())  # Total number of NaNs in the entire data

# Clean data
data_lending_ny_clean = data_lending_ny_short.dropna(axis='index')
print(data_lending_ny_clean.shape)
print(data_lending_ny_clean['response'].value_counts())

# Check correlations
correlation_matrix = data_lending_ny_clean.corr()
print(correlation_matrix['response'])

# Create folder for saved data
save_dir = "../saved_data"
os.makedirs(save_dir, exist_ok=True)