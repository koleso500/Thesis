import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from safeai_files.utils import plot_model_curves

# Load values
file_path_random = os.path.join("../saved_data", "final_results_random_ny_article.json")
file_path_rf = os.path.join("../saved_data", "final_results_rf_ny_article.json")
file_path_xgb = os.path.join("../saved_data", "final_results_xgb_ny_article.json")
with open(file_path_random, "r", encoding="utf-8") as file:
    data_random = json.load(file)
with open(file_path_rf, "r", encoding="utf-8") as file:
    data_rf = json.load(file)
with open(file_path_xgb, "r", encoding="utf-8") as file:
    data_xgb = json.load(file)

x_random = data_random["x_final"]
y_random = data_random["y_final"]
z_random = data_random["z_final"]
x_rf = data_rf["x_final"]
y_rf = data_rf["y_final"]
z_rf = data_rf["z_final"]
x_xgb = data_xgb["x_final"]
y_xgb = data_xgb["y_final"]
z_xgb = data_xgb["z_final"]

# Differences
x_rf_r = (np.array(x_rf) - np.array(x_random)).tolist()
y_rf_r = (np.array(y_rf) - np.array(y_random)).tolist()
z_rf_r = (np.array(z_rf) - np.array(z_random)).tolist()

x_xgb_r = (np.array(x_xgb) - np.array(x_random)).tolist()
y_xgb_r = (np.array(y_xgb) - np.array(y_random)).tolist()
z_xgb_r = (np.array(z_xgb) - np.array(z_random)).tolist()

# All curves for RF
x_rga = np.linspace(0, 1, 20)
plot_model_curves(x_rga,[x_rf, y_rf, z_rf], model_name="RF", title="Random Forest Curves (New York Article)")

# All curves for XGB
plot_model_curves(x_rga,[x_xgb, y_xgb, z_xgb], model_name="XGB", title="XGBoosting Curves (New York Article)")

# All curves for Random
plot_model_curves(x_rga,[x_random, y_random, z_random], model_name="Random",title="Random Classifier Curves (New York Article)")

# All curves for difference RF and Random
plot_model_curves(x_rga,[x_rf_r, y_rf_r, z_rf_r], model_name="Random", prefix="Difference",
                  title="RF and Random Curves Difference (New York Article)")

# All curves for difference XGB and Random
plot_model_curves(x_rga,[x_xgb_r, y_xgb_r, z_xgb_r], model_name="Random", prefix="Difference",
                  title="XGB and Random Curves Difference (New York Article)")

plt.show()

# Values and Volume
rgas_random = np.array(x_random)
rges_random = np.array(y_random)
rgrs_random = np.array(z_random)
rgas_rf = np.array(x_rf)
rges_rf = np.array(y_rf)
rgrs_rf = np.array(z_rf)
rgas_xgb = np.array(x_xgb)
rges_xgb = np.array(y_xgb)
rgrs_xgb = np.array(z_xgb)

# Scalar fields, matrix of initial values
rga_r, rge_r, rgr_r = np.meshgrid(rgas_random, rges_random, rgrs_random, indexing='ij')
rga_rf, rge_rf, rgr_rf = np.meshgrid(rgas_rf, rges_rf, rgrs_rf, indexing='ij')
rga_xgb, rge_xgb, rgr_xgb = np.meshgrid(rgas_xgb, rges_xgb, rgrs_xgb, indexing='ij')

# Histogram of the arithmetic mean metric RF
values_a_r = (rga_r + rge_r + rgr_r) / 3
values_a_rf = (rga_rf + rge_rf + rgr_rf) / 3

# Normalize
values_a_r_norm = values_a_r / np.max(values_a_r)
values_a_rf_norm = values_a_rf / np.max(values_a_rf)

# Total sum and number of elements
total_sum_r = np.sum(values_a_r_norm)
total_sum_rf = np.sum(values_a_rf_norm)
num_elements_r = values_a_r_norm.size
num_elements_rf = values_a_rf_norm.size

# Normalized volume
normalized_volume_r = total_sum_r / num_elements_r
normalized_volume_rf = total_sum_rf / num_elements_rf
print("Normalized volume Random Classifier:", normalized_volume_r)
print("Normalized volume Random Forest:", normalized_volume_rf)

# Plot
flat_vals_a_r = values_a_r_norm.flatten().tolist()
flat_vals_a_rf = values_a_rf_norm.flatten()

counts_r, bins_r = np.histogram(flat_vals_a_r, bins=60)
counts_rf, bins_rf = np.histogram(flat_vals_a_rf, bins=60)

max_count = max(counts_r.max(), counts_rf.max())
counts_norm_r = counts_r / max_count
counts_norm_rf = counts_rf / max_count
bin_centers_r = (bins_r[:-1] + bins_r[1:]) / 2
bin_centers_rf = (bins_rf[:-1] + bins_rf[1:]) / 2

plt.figure(figsize=(10, 6))
plt.bar(bin_centers_r, counts_norm_r, width=(bins_r[1] - bins_r[0]), alpha=0.7, label='Random Classifier')
plt.bar(bin_centers_rf, counts_norm_rf, width=(bins_rf[1] - bins_rf[0]), alpha=0.7, label='Random Forest Model')
plt.xlabel('Normalized Arithmetic Mean')
plt.ylabel('Normalized Counts')
plt.title('Histogram of Normalized Arithmetic Mean Values(RF)')
plt.grid(True)
plt.legend()
plt.show()

# Histogram of the arithmetic mean metric XGB
values_a_xgb = (rga_xgb + rge_xgb + rgr_xgb) / 3

# Normalize
values_a_xgb_norm = values_a_xgb / np.max(values_a_xgb)

# Total sum and number of elements
total_sum_xgb = np.sum(values_a_xgb_norm)
num_elements_xgb = values_a_xgb_norm.size

# Normalized volume
normalized_volume_xgb = total_sum_xgb / num_elements_xgb
print("Normalized volume XGBoosting:", normalized_volume_xgb)

# Plot
flat_vals_a_xgb = values_a_xgb_norm.flatten()
counts_xgb, bins_xgb = np.histogram(flat_vals_a_xgb, bins=60)

max_count = max(counts_r.max(), counts_xgb.max())
counts_norm_xgb = counts_xgb / max_count
bin_centers_xgb = (bins_rf[:-1] + bins_rf[1:]) / 2

plt.figure(figsize=(10, 6))
plt.bar(bin_centers_r, counts_norm_r, width=(bins_r[1] - bins_r[0]), alpha=0.7, label='Random Classifier')
plt.bar(bin_centers_xgb, counts_norm_xgb, width=(bins_rf[1] - bins_rf[0]), alpha=0.7, label='XGB Model')
plt.xlabel('Normalized Arithmetic Mean')
plt.ylabel('Normalized Counts')
plt.title('Histogram of Normalized Arithmetic Mean Values(XGB)')
plt.grid(True)
plt.legend()
plt.show()

# Histogram of the geometric mean metric
values_gm_r = (rga_r ** 2 + rge_r ** 2 + rgr_r ** 2) ** (1/2)
values_gm_rf = (rga_rf ** 2 + rge_rf ** 2 + rgr_rf ** 2) ** (1/2)
volume_gm_r = np.sum(values_gm_r)
volume_gm_rf = np.sum(values_gm_rf)
print(volume_gm_rf)

flat_vals_gm_r = values_gm_r.flatten().tolist()
flat_vals_gm_r_sample = random.sample(flat_vals_gm_r, 800)
flat_vals_gm_rf = values_gm_rf.flatten()
plt.figure(figsize=(10, 6))
# plt.hist(flat_vals_gm_r_sample, bins=60, alpha=0.6, label='Random Classifier')
plt.hist(flat_vals_gm_rf, bins=60, label='Random Forest Model', density=True, stacked=True)
plt.xlabel('Geometric Mean')
plt.ylabel('Count')
plt.title('Histogram of Geometric Mean Values(RF)')
plt.grid(True)

values_gm_xgb = (rga_xgb ** 2 + rge_xgb ** 2 + rgr_xgb ** 2) ** (1/2)
volume_gm_xgb = np.sum(values_gm_xgb)
print(volume_gm_xgb)

flat_vals_gm_xgb = values_gm_xgb.flatten()
plt.figure(figsize=(10, 6))
# plt.hist(flat_vals_gm_r_sample, bins=60, alpha=0.6, label='Random Classifier')
plt.hist(flat_vals_gm_xgb, bins=60, label='XGB', density=True, stacked=True)
plt.xlabel('Geometric Mean')
plt.ylabel('Count')
plt.title('Histogram of Geometric Mean Values(XGB)')
plt.grid(True)
plt.show()

# Histogram of the harmonic mean metric
values_hm_r = 3.0 / (1.0 / rga_r + 1.0 / rge_r + 1.0 / rgr_r)
values_hm_rf = 3.0 / (1.0 / rga_rf + 1.0 / rge_rf + 1.0 / rgr_rf)

flat_vals_hm_r = values_hm_r.flatten().tolist()
flat_vals_hm_r_sample = random.sample(flat_vals_hm_r, 800)
flat_vals_hm_rf = values_hm_rf.flatten()
plt.figure(figsize=(10, 6))
plt.hist(flat_vals_hm_r_sample, bins=60, alpha=0.7, label='Random Classifier')
plt.hist(flat_vals_hm_rf, bins=60, label='Random Forest Model')
plt.xlabel('Harmonic Mean')
plt.ylabel('Count')
plt.title('Histogram of Harmonic Mean Values')
plt.grid(True)
plt.show()