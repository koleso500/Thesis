import json
import matplotlib.pyplot as plt
import numpy as np
import os

from safeai_files.utils import plot_mean_histogram, plot_model_curves, plot_metric_distribution, plot_diff_mean_histogram

# Load values
file_path_lr = os.path.join("../saved_data", "final_results_lr_ca.json")
file_path_rf = os.path.join("../saved_data", "final_results_rf_ca.json")
file_path_xgb = os.path.join("../saved_data", "final_results_xgb_ca.json")
file_path_stacked = os.path.join("../saved_data", "final_results_stacked_ca.json")
file_path_voting = os.path.join("../saved_data", "final_results_voting_ca.json")
file_path_neural = os.path.join("../saved_data", "final_results_neural_ca.json")
file_path_random = os.path.join("../saved_data", "final_results_random_ca.json")

with open(file_path_lr, "r", encoding="utf-8") as file:
    data_lr = json.load(file)
with open(file_path_rf, "r", encoding="utf-8") as file:
    data_rf = json.load(file)
with open(file_path_xgb, "r", encoding="utf-8") as file:
    data_xgb = json.load(file)
with open(file_path_stacked, "r", encoding="utf-8") as file:
    data_stacked = json.load(file)
with open(file_path_voting, "r", encoding="utf-8") as file:
    data_voting = json.load(file)
with open(file_path_neural, "r", encoding="utf-8") as file:
    data_neural = json.load(file)
with open(file_path_random, "r", encoding="utf-8") as file:
    data_random = json.load(file)

x_lr = data_lr["x_final"]
y_lr = data_lr["y_final"]
z_lr = data_lr["z_final"]

x_rf = data_rf["x_final"]
y_rf = data_rf["y_final"]
z_rf = data_rf["z_final"]

x_xgb = data_xgb["x_final"]
y_xgb = data_xgb["y_final"]
z_xgb = data_xgb["z_final"]

x_stacked = data_stacked["x_final"]
y_stacked = data_stacked["y_final"]
z_stacked = data_stacked["z_final"]

x_voting = data_voting["x_final"]
y_voting = data_voting["y_final"]
z_voting = data_voting["z_final"]

x_neural = data_neural["x_final"]
y_neural = data_neural["y_final"]
z_neural = data_neural["z_final"]

x_random = data_random["x_final"]
y_random = data_random["y_final"]
z_random = data_random["z_final"]

# Differences
x_lr_r = (np.array(x_lr) - np.array(x_random)).tolist()
y_lr_r = (np.array(y_lr) - np.array(y_random)).tolist()
z_lr_r = (np.array(z_lr) - np.array(z_random)).tolist()

x_rf_r = (np.array(x_rf) - np.array(x_random)).tolist()
y_rf_r = (np.array(y_rf) - np.array(y_random)).tolist()
z_rf_r = (np.array(z_rf) - np.array(z_random)).tolist()

x_xgb_r = (np.array(x_xgb) - np.array(x_random)).tolist()
y_xgb_r = (np.array(y_xgb) - np.array(y_random)).tolist()
z_xgb_r = (np.array(z_xgb) - np.array(z_random)).tolist()

x_stacked_r = (np.array(x_stacked) - np.array(x_random)).tolist()
y_stacked_r = (np.array(y_stacked) - np.array(y_random)).tolist()
z_stacked_r = (np.array(z_stacked) - np.array(z_random)).tolist()

x_voting_r = (np.array(x_voting) - np.array(x_random)).tolist()
y_voting_r = (np.array(y_voting) - np.array(y_random)).tolist()
z_voting_r = (np.array(z_voting) - np.array(z_random)).tolist()

x_neural_r = (np.array(x_neural) - np.array(x_random)).tolist()
y_neural_r = (np.array(y_neural) - np.array(y_random)).tolist()
z_neural_r = (np.array(z_neural) - np.array(z_random)).tolist()

# All curves for LR
x_rga = np.linspace(0, 1, len(y_random))
plot_model_curves(x_rga,[x_lr, y_lr, z_lr], model_name="LR", title="Logistic Regression Curves (California)")

# All curves for RF
plot_model_curves(x_rga,[x_rf, y_rf, z_rf], model_name="RF", title="Random Forest Curves (California)")

# All curves for XGB
plot_model_curves(x_rga,[x_xgb, y_xgb, z_xgb], model_name="XGB", title="XGBoosting Curves (California)")

# All curves for SE
plot_model_curves(x_rga,[x_stacked, y_stacked, z_stacked], model_name="SE", title="Stacked Ensemble Curves (California)")

# All curves for VE
plot_model_curves(x_rga,[x_voting, y_voting, z_voting], model_name="VE", title="Voting Ensemble Curves (California)")

# All curves for NN
plot_model_curves(x_rga,[x_neural, y_neural, z_neural], model_name="NN", title="Neural Network Curves (California)")

# All curves for Random
plot_model_curves(x_rga,[x_random, y_random, z_random], model_name="Random",title="Random Classifier Curves (California)")

# All curves for difference LR and Random
plot_model_curves(x_rga,[x_lr_r, y_lr_r, z_lr_r], model_name="Random", prefix="Difference",
                  title="LR and Random Curves Difference (California)")

# All curves for difference RF and Random
plot_model_curves(x_rga,[x_rf_r, y_rf_r, z_rf_r], model_name="Random", prefix="Difference",
                  title="RF and Random Curves Difference (California)")

# All curves for difference XGB and Random
plot_model_curves(x_rga,[x_xgb_r, y_xgb_r, z_xgb_r], model_name="Random", prefix="Difference",
                  title="XGB and Random Curves Difference (California)")

# All curves for difference SE and Random
plot_model_curves(x_rga,[x_stacked_r, y_stacked_r, z_stacked_r], model_name="Random", prefix="Difference",
                  title="SE and Random Curves Difference (California)")

# All curves for difference VE and Random
plot_model_curves(x_rga,[x_voting_r, y_voting_r, z_voting_r], model_name="Random", prefix="Difference",
                  title="VE and Random Curves Difference (California)")

# All curves for difference NN and Random
plot_model_curves(x_rga,[x_neural_r, y_neural_r, z_neural_r], model_name="Random", prefix="Difference",
                  title="NN and Random Curves Difference (California)")

plt.show()

# Values and Volume
rgas_lr = np.array(x_lr)
rges_lr = np.array(y_lr)
rgrs_lr = np.array(z_lr)

rgas_rf = np.array(x_rf)
rges_rf = np.array(y_rf)
rgrs_rf = np.array(z_rf)

rgas_xgb = np.array(x_xgb)
rges_xgb = np.array(y_xgb)
rgrs_xgb = np.array(z_xgb)

rgas_se = np.array(x_stacked)
rges_se = np.array(y_stacked)
rgrs_se = np.array(z_stacked)

rgas_ve = np.array(x_voting)
rges_ve = np.array(y_voting)
rgrs_ve = np.array(z_voting)

rgas_nn = np.array(x_neural)
rges_nn = np.array(y_neural)
rgrs_nn = np.array(z_neural)

rgas_random = np.array(x_random)
rges_random = np.array(y_random)
rgrs_random = np.array(z_random)

# Scalar fields, matrix of initial values
rga_lr, rge_lr, rgr_lr = np.meshgrid(rgas_lr, rges_lr, rgrs_lr, indexing='ij')
rga_rf, rge_rf, rgr_rf = np.meshgrid(rgas_rf, rges_rf, rgrs_rf, indexing='ij')
rga_xgb, rge_xgb, rgr_xgb = np.meshgrid(rgas_xgb, rges_xgb, rgrs_xgb, indexing='ij')
rga_se, rge_se, rgr_se = np.meshgrid(rgas_se, rges_se, rgrs_se, indexing='ij')
rga_ve, rge_ve, rgr_ve = np.meshgrid(rgas_ve, rges_ve, rgrs_ve, indexing='ij')
rga_nn, rge_nn, rgr_nn = np.meshgrid(rgas_nn, rges_nn, rgrs_nn, indexing='ij')
rga_r, rge_r, rgr_r = np.meshgrid(rgas_random, rges_random, rgrs_random, indexing='ij')

# Means
models = [
    ((rga_lr,  rge_lr,  rgr_lr),  "Logistic Regression", "Logistic Regression"),
    ((rga_rf,  rge_rf,  rgr_rf),  "Random Forest", "Random Forest Model"),
    ((rga_xgb, rge_xgb, rgr_xgb), "XGBoosting", "XGB Model"),
    ((rga_se,  rge_se,  rgr_se),  "Stacked Ensemble", "Stacked Ensemble Model"),
    ((rga_ve,  rge_ve,  rgr_ve),  "Voting Ensemble", "Voting Ensemble Model"),
    ((rga_nn,  rge_nn,  rgr_nn),  "Neural Network", "Neural Network Model"),
    ((rga_r,   rge_r,   rgr_r),   "Random Classifier", "Random Classifier"),
]

# All arithmetic mean histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models:
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="arithmetic"
    )
plt.show()

# All geometric mean histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models:
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="geometric"
    )
plt.show()

# All quadratic mean histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models:
    plot_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="quadratic"
    )
plt.show()

# Differences Means
# Values
rga_d_lr = np.array(x_lr_r)
rge_d_lr = np.array(y_lr_r)
rgr_d_lr = np.array(z_lr_r)

rga_d_rf = np.array(x_rf_r)
rge_d_rf = np.array(y_rf_r)
rgr_d_rf = np.array(z_rf_r)

rga_d_xgb = np.array(x_xgb_r)
rge_d_xgb = np.array(y_xgb_r)
rgr_d_xgb = np.array(z_xgb_r)

rga_d_se = np.array(x_stacked_r)
rge_d_se = np.array(y_stacked_r)
rgr_d_se = np.array(z_stacked_r)

rga_d_ve = np.array(x_voting_r)
rge_d_ve = np.array(y_voting_r)
rgr_d_ve = np.array(z_voting_r)

rga_d_nn = np.array(x_neural_r)
rge_d_nn = np.array(y_neural_r)
rgr_d_nn = np.array(z_neural_r)

rga_lr_d, rge_lr_d, rgr_lr_d = np.meshgrid(rga_d_lr, rge_d_lr, rgr_d_lr, indexing='ij')
rga_rf_d, rge_rf_d, rgr_rf_d = np.meshgrid(rga_d_rf, rge_d_rf, rgr_d_rf, indexing='ij')
rga_xgb_d, rge_xgb_d, rgr_xgb_d = np.meshgrid(rga_d_xgb, rge_d_xgb, rgr_d_xgb, indexing='ij')
rga_se_d, rge_se_d, rgr_se_d = np.meshgrid(rga_d_se, rge_d_se, rgr_d_se, indexing='ij')
rga_ve_d, rge_ve_d, rgr_ve_d = np.meshgrid(rga_d_ve, rge_d_ve, rgr_d_ve, indexing='ij')
rga_nn_d, rge_nn_d, rgr_nn_d = np.meshgrid(rga_d_nn, rge_d_nn, rgr_d_nn, indexing='ij')

models_diff = [
    ((rga_lr_d,  rge_lr_d,  rgr_lr_d),  "Logistic Regression", "Logistic Regression"),
    ((rga_rf_d,  rge_rf_d,  rgr_rf_d),  "Random Forest", "Random Forest Model"),
    ((rga_xgb_d, rge_xgb_d, rgr_xgb_d), "XGBoosting", "XGB Model"),
    ((rga_se_d,  rge_se_d,  rgr_se_d),  "Stacked Ensemble", "Stacked Ensemble Model"),
    ((rga_ve_d,  rge_ve_d,  rgr_ve_d),  "Voting Ensemble", "Voting Ensemble Model"),
    ((rga_nn_d,  rge_nn_d,  rgr_nn_d),  "Neural Network", "Neural Network Model"),
]

# All arithmetic mean differences histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models_diff:
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="arithmetic"
    )
plt.show()

# All geometric mean differences histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models_diff:
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="geometric"
    )
plt.show()

# All quadratic mean differences histograms
for (rga_var, rge_var, rgr_var), model_name, bar_label in models_diff:
    plot_diff_mean_histogram(
        rga_var, rge_var, rgr_var,
        model_name=model_name,
        bar_label=bar_label,
        mean_type="quadratic"
    )
plt.show()

# Slope Arithmetic
def slope_arithmetic(x, y, z):
    dx = [x[i] - x[i+1] for i in range(len(x)-1)]
    dy = [y[i] - y[i+1] for i in range(len(y)-1)]
    dz = [z[i] - z[i+1] for i in range(len(z)-1)]
    rgas = np.array(dx)
    rges = np.array(dy)
    rgrs = np.array(dz)
    rga, rge, rgr = np.meshgrid(rgas, rges, rgrs, indexing='ij')
    results = (rga + rge + rgr) / 3
    return results

# Histogram of the slope product metric LR
values_sl_lr = slope_arithmetic(x_lr, y_lr, z_lr)
plot_metric_distribution(
    metric_values=values_sl_lr,
    print_label="Mean volume Slope Product Logistic Regression",
    xlabel="Normalized Slope Product",
    title="Histogram of Normalized Slope Product Values (Logistic Regression)",
    bar_label="Logistic Regression"
)

# Histogram of the slope product metric RF
values_sl_rf = slope_arithmetic(x_rf, y_rf, z_rf)
plot_metric_distribution(
    metric_values=values_sl_rf,
    print_label="Mean volume Slope Product Random Forest",
    xlabel="Normalized Slope Product",
    title="Histogram of Normalized Slope Product Values (Random Forest)",
    bar_label="Random Forest"
)

# Histogram of the slope product metric XGB
values_sl_xgb = slope_arithmetic(x_xgb, y_xgb, z_xgb)
plot_metric_distribution(
    metric_values=values_sl_xgb,
    print_label="Mean volume Slope Product XGBoosting",
    xlabel="Normalized Slope Product",
    title="Histogram of Normalized Slope Product Values (XGBoosting)",
    bar_label="XGBoosting"
)

# Histogram of the slope product metric SE
values_sl_se = slope_arithmetic(x_stacked, y_stacked, z_stacked)
plot_metric_distribution(
    metric_values=values_sl_se,
    print_label="Mean volume Slope Product Stacked Ensemble",
    xlabel="Normalized Slope Product",
    title="Histogram of Normalized Slope Product Values (Stacked Ensemble)",
    bar_label="Stacked Ensemble"
)

# Histogram of the slope product metric VE
values_sl_ve = slope_arithmetic(x_voting, y_voting, z_voting)
plot_metric_distribution(
    metric_values=values_sl_ve,
    print_label="Mean volume Slope Product Voting Ensemble",
    xlabel="Normalized Slope Product",
    title="Histogram of Normalized Slope Product Values (Voting Ensemble)",
    bar_label="Voting Ensemble"
)

# Histogram of the slope product metric NN
values_sl_nn = slope_arithmetic(x_neural, y_neural, z_neural)
plot_metric_distribution(
    metric_values=values_sl_nn,
    print_label="Mean volume Slope Product Neural Network",
    xlabel="Normalized Slope Product",
    title="Histogram of Normalized Slope Product Values (Neural Network)",
    bar_label="Neural Network"
)

# Histogram of the slope product metric Random
values_sl_r = slope_arithmetic(x_random, y_random, z_random)
plot_metric_distribution(
    metric_values=values_sl_r,
    print_label="Mean volume Slope Product Random Classifier",
    xlabel="Normalized Slope Product",
    title="Histogram of Normalized Slope Product Values (Random Classifier)",
    bar_label="Random Classifier"
)

plt.show()

# Hypervolume approach
def hypervolume(x, y, z):
    v1 = np.array(x)
    v2 = np.array(y)
    v3 = np.array(z)

    # Construct Gram matrix
    g = np.array([
        [np.dot(v1, v1), np.dot(v1, v2), np.dot(v1, v3)],
        [np.dot(v2, v1), np.dot(v2, v2), np.dot(v2, v3)],
        [np.dot(v3, v1), np.dot(v3, v2), np.dot(v3, v3)]
    ])

    # Hypervolume
    volume = np.sqrt(np.linalg.det(g))

    return volume

# Hypervolume LR
volume_lr = hypervolume(rgas_lr, rges_lr, rgrs_lr)
print(f"Hypervolume LR: {volume_lr:.3f}")

# Hypervolume RF
volume_rf = hypervolume(rgas_rf, rges_rf, rgrs_rf)
print(f"Hypervolume RF: {volume_rf:.3f}")

# Hypervolume XGB
volume_xgb = hypervolume(rgas_xgb, rges_xgb, rgrs_xgb)
print(f"Hypervolume XGB: {volume_xgb:.3f}")

# Hypervolume SE
volume_se = hypervolume(rgas_se, rges_se, rgrs_se)
print(f"Hypervolume SE: {volume_se:.3f}")

# Hypervolume VE
volume_ve = hypervolume(rgas_ve, rges_ve, rgrs_ve)
print(f"Hypervolume VE: {volume_ve:.3f}")

# Hypervolume NN
volume_nn = hypervolume(rgas_nn, rges_nn, rgrs_nn)
print(f"Hypervolume NN: {volume_nn:.3f}")

# Hypervolume R
volume_r = hypervolume(rgas_random, rges_random, rgrs_random)
print(f"Hypervolume Random: {volume_r:.3f}")