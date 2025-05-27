import json
import matplotlib.pyplot as plt
import numpy as np
import os

from safeai_files.utils import plot_model_curves, plot_metric_distribution, plot_metric_distribution_diff

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

# Histogram of the arithmetic mean metric LR
values_a_lr = (rga_lr + rge_lr + rgr_lr) / 3
plot_metric_distribution(
    metric_values=values_a_lr,
    print_label="Mean volume Arithmetic Logistic Regression",
    xlabel="Normalized Arithmetic Mean",
    title="Histogram of Normalized Arithmetic Mean Values (Logistic Regression)",
    bar_label="Logistic Regression"
)

# Histogram of the arithmetic mean metric RF
values_a_rf = (rga_rf + rge_rf + rgr_rf) / 3
plot_metric_distribution(
    metric_values=values_a_rf,
    print_label="Mean volume Arithmetic Random Forest",
    xlabel="Normalized Arithmetic Mean",
    title="Histogram of Normalized Arithmetic Mean Values (RF)",
    bar_label="Random Forest Model"
)

# Histogram of the arithmetic mean metric XGB
values_a_xgb = (rga_xgb + rge_xgb + rgr_xgb) / 3
plot_metric_distribution(
    metric_values=values_a_xgb,
    print_label="Mean volume Arithmetic XGBoosting",
    xlabel="Normalized Arithmetic Mean",
    title="Histogram of Normalized Arithmetic Mean Values (XGB)",
    bar_label="XGB Model"
)

# Histogram of the arithmetic mean metric SE
values_a_se = (rga_se + rge_se + rgr_se) / 3
plot_metric_distribution(
    metric_values=values_a_se,
    print_label="Mean volume Arithmetic Stacked Ensemble",
    xlabel="Normalized Arithmetic Mean",
    title="Histogram of Normalized Arithmetic Mean Values (Stacked Ensemble)",
    bar_label="Stacked Ensemble Model"
)

# Histogram of the arithmetic mean metric VE
values_a_ve = (rga_ve + rge_ve + rgr_ve) / 3
plot_metric_distribution(
    metric_values=values_a_ve,
    print_label="Mean volume Arithmetic Voting Ensemble",
    xlabel="Normalized Arithmetic Mean",
    title="Histogram of Normalized Arithmetic Mean Values (Voting Ensemble)",
    bar_label="Voting Ensemble Model"
)

# Histogram of the arithmetic mean metric NN
values_a_nn = (rga_nn + rge_nn + rgr_nn) / 3
plot_metric_distribution(
    metric_values=values_a_nn,
    print_label="Mean volume Arithmetic Neural Network",
    xlabel="Normalized Arithmetic Mean",
    title="Histogram of Normalized Arithmetic Mean Values (Neural Network)",
    bar_label="Neural Network Model"
)

# Histogram of the arithmetic mean metric Random
values_a_r = (rga_r + rge_r + rgr_r) / 3
plot_metric_distribution(
    metric_values=values_a_r,
    print_label="Mean volume Arithmetic Random Classifier",
    xlabel="Normalized Arithmetic Mean",
    title="Histogram of Normalized Arithmetic Mean Values (Random)",
    bar_label="Random Classifier"
)

plt.show()

# Histogram of the geometric mean metric LR (1/3)
values_g_lr = (rga_lr * rge_lr * rgr_lr) ** (1/3)
plot_metric_distribution(
    metric_values=values_g_lr,
    print_label="Mean volume Geometric Logistic Regression",
    xlabel="Normalized Geometric Mean (1/3)",
    title="Histogram of Normalized Geometric Mean (1/3) Values (Logistic Regression)",
    bar_label="Logistic Regression"
)

# Histogram of the geometric mean metric RF (1/3)
values_g_rf = (rga_rf * rge_rf * rgr_rf) ** (1/3)
plot_metric_distribution(
    metric_values=values_g_rf,
    print_label="Mean volume Geometric Random Forest",
    xlabel="Normalized Geometric Mean (1/3)",
    title="Histogram of Normalized Geometric Mean (1/3) Values (RF)",
    bar_label="Random Forest Model"
)

# Histogram of the geometric mean metric XGB (1/3)
values_g_xgb = (rga_xgb * rge_xgb * rgr_xgb) ** (1/3)
plot_metric_distribution(
    metric_values=values_g_xgb,
    print_label="Mean volume Geometric XGBoosting",
    xlabel="Normalized Geometric Mean (1/3)",
    title="Histogram of Normalized Geometric Mean (1/3) Values (XGB)",
    bar_label="XGB Model"
)

# Histogram of the geometric mean metric SE (1/3)
values_g_se = (rga_se * rge_se * rgr_se) ** (1/3)
plot_metric_distribution(
    metric_values=values_g_se,
    print_label="Mean volume Geometric Stacked Ensemble",
    xlabel="Normalized Geometric Mean (1/3)",
    title="Histogram of Normalized Geometric Mean (1/3) Values (Stacked Ensemble)",
    bar_label="Stacked Ensemble Model"
)

# Histogram of the geometric mean metric VE (1/3)
values_g_ve = (rga_ve * rge_ve * rgr_ve) ** (1/3)
plot_metric_distribution(
    metric_values=values_g_ve,
    print_label="Mean volume Geometric Voting Ensemble",
    xlabel="Normalized Geometric Mean (1/3)",
    title="Histogram of Normalized Geometric Mean (1/3) Values (Voting Ensemble)",
    bar_label="Voting Ensemble Model"
)

# Histogram of the geometric mean metric NN (1/3)
values_g_nn = (rga_nn * rge_nn * rgr_nn) ** (1/3)
plot_metric_distribution(
    metric_values=values_g_nn,
    print_label="Mean volume Geometric Neural Network",
    xlabel="Normalized Geometric Mean (1/3)",
    title="Histogram of Normalized Geometric Mean (1/3) Values (Neural Network)",
    bar_label="Neural Network Model"
)

# Histogram of the geometric mean metric Random (1/3)
values_g_r = (rga_r * rge_r * rgr_r) ** (1/3)
plot_metric_distribution(
    metric_values=values_g_r,
    print_label="Mean volume Geometric Random Classifier",
    xlabel="Normalized Geometric Mean (1/3)",
    title="Histogram of Normalized Geometric Mean (1/3) Values (Random)",
    bar_label="Random Classifier"
)

plt.show()

# Histogram of the Quadratic Mean (RMS) metric LR
values_rms_lr = ((rga_lr ** 2 + rge_lr ** 2 + rgr_lr ** 2) / 3) ** (1/2)
plot_metric_distribution(
    metric_values=values_rms_lr,
    print_label="Mean volume Quadratic Mean (RMS) Logistic Regression",
    xlabel="Normalized Quadratic Mean (RMS)",
    title="Histogram of Normalized Quadratic Mean (RMS) Values (Logistic Regression)",
    bar_label="Logistic Regression"
)

# Histogram of the Quadratic Mean (RMS) metric RF
values_rms_rf = ((rga_rf ** 2 + rge_rf ** 2 + rgr_rf ** 2) / 3) ** (1/2)
plot_metric_distribution(
    metric_values=values_rms_rf,
    print_label="Mean volume Quadratic Mean (RMS) Random Forest",
    xlabel="Normalized Quadratic Mean (RMS)",
    title="Histogram of Normalized Quadratic Mean (RMS) Values (RF)",
    bar_label="Random Forest Model"
)

# Histogram of the Quadratic Mean (RMS) metric XGB
values_rms_xgb = ((rga_xgb ** 2 + rge_xgb ** 2 + rgr_xgb ** 2) / 3) ** (1/2)
plot_metric_distribution(
    metric_values=values_rms_xgb,
    print_label="Mean volume Quadratic Mean (RMS) XGBoosting",
    xlabel="Normalized Quadratic Mean (RMS)",
    title="Histogram of Normalized Quadratic Mean (RMS) Values (XGB)",
    bar_label="XGB Model"
)

# Histogram of the Quadratic Mean (RMS) metric SE
values_rms_se = ((rga_se ** 2 + rge_se ** 2 + rgr_se ** 2) / 3) ** (1/2)
plot_metric_distribution(
    metric_values=values_rms_se,
    print_label="Mean volume Quadratic Mean (RMS) Stacked Ensemble",
    xlabel="Normalized Quadratic Mean (RMS)",
    title="Histogram of Normalized Quadratic Mean (RMS) Values (Stacked Ensemble)",
    bar_label="Stacked Ensemble Model"
)

# Histogram of the Quadratic Mean (RMS) metric VE
values_rms_ve = ((rga_ve ** 2 + rge_ve ** 2 + rgr_ve ** 2) / 3) ** (1/2)
plot_metric_distribution(
    metric_values=values_rms_ve,
    print_label="Mean volume Quadratic Mean (RMS) Voting Ensemble",
    xlabel="Normalized Quadratic Mean (RMS)",
    title="Histogram of Normalized Quadratic Mean (RMS) Values (Voting Ensemble)",
    bar_label="Voting Ensemble Model"
)

# Histogram of the Quadratic Mean (RMS) metric NN
values_rms_nn = ((rga_nn ** 2 + rge_nn ** 2 + rgr_nn ** 2) / 3) ** (1/2)
plot_metric_distribution(
    metric_values=values_rms_nn,
    print_label="Mean volume Quadratic Mean (RMS) Neural Network",
    xlabel="Normalized Quadratic Mean (RMS)",
    title="Histogram of Normalized Quadratic Mean (RMS) Values (Neural Network)",
    bar_label="Neural Network Model"
)

# Histogram of the Quadratic Mean (RMS) metric Random
values_rms_r = ((rga_r ** 2 + rge_r ** 2 + rgr_r ** 2) / 3) ** (1/2)
plot_metric_distribution(
    metric_values=values_rms_r,
    print_label="Mean volume Quadratic Mean (RMS) Random Classifier",
    xlabel="Normalized Quadratic Mean (RMS)",
    title="Histogram of Normalized Quadratic Mean (RMS) Values (Random)",
    bar_label="Random Classifier"
)

plt.show()

# Slope Arithmetic
def slope_arithmetic(x, y, z):
    dx = [x[i] - x[i+1] for i in range(len(x)-1)]
    dy = [y[i+1] - y[i] for i in range(len(y)-1)]
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

# Differences Plots Arithmetic Mean
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

values_diff_lr = (rga_lr_d + rge_lr_d + rgr_lr_d) / 3
values_diff_rf = (rga_rf_d + rge_rf_d + rgr_rf_d) / 3
values_diff_xgb = (rga_xgb_d + rge_xgb_d + rgr_xgb_d) / 3
values_diff_se = (rga_se_d + rge_se_d + rgr_se_d) / 3
values_diff_ve = (rga_ve_d + rge_ve_d + rgr_ve_d) / 3
values_diff_nn = (rga_nn_d + rge_nn_d + rgr_nn_d) / 3


plot_metric_distribution_diff(
    metric_values=values_diff_lr,
    print_label="Difference Arithmetic Logistic Regression",
    xlabel="Difference Arithmetic Mean",
    title="Histogram of Difference Arithmetic Mean Values (Logistic Regression)",
    bar_label="Logistic Regression"
)

plot_metric_distribution_diff(
    metric_values=values_diff_rf,
    print_label="Difference Arithmetic Random Forest",
    xlabel="Difference Arithmetic Mean",
    title="Histogram of Difference Arithmetic Mean Values (Random Forest)",
    bar_label="Random Forest"
)

plot_metric_distribution_diff(
    metric_values=values_diff_xgb,
    print_label="Difference Arithmetic XGBoosting",
    xlabel="Difference Arithmetic Mean",
    title="Histogram of Difference Arithmetic Mean Values (XGBoosting)",
    bar_label="XGBoosting"
)

plot_metric_distribution_diff(
    metric_values=values_diff_se,
    print_label="Difference Arithmetic Stacked Ensemble",
    xlabel="Difference Arithmetic Mean",
    title="Histogram of Difference Arithmetic Mean Values (Stacked Ensemble)",
    bar_label="Stacked Ensemble"
)

plot_metric_distribution_diff(
    metric_values=values_diff_ve,
    print_label="Difference Arithmetic Voting Ensemble",
    xlabel="Difference Arithmetic Mean",
    title="Histogram of Difference Arithmetic Mean Values (Voting Ensemble)",
    bar_label="Voting Ensemble"
)

plot_metric_distribution_diff(
    metric_values=values_diff_nn,
    print_label="Difference Arithmetic Neural Network",
    xlabel="Difference Arithmetic Mean",
    title="Histogram of Difference Arithmetic Mean Values (Neural Network)",
    bar_label="Neural Network"
)

plt.show()

# Differences Plots Geometric Mean (1/3)
values_diff_g_lr = np.cbrt(rga_lr_d * rge_lr_d * rgr_lr_d)
values_diff_g_rf = np.cbrt(rga_rf_d * rge_rf_d * rgr_rf_d)
values_diff_g_xgb = np.cbrt(rga_xgb_d * rge_xgb_d * rgr_xgb_d)
values_diff_g_se = np.cbrt(rga_se_d * rge_se_d * rgr_se_d)
values_diff_g_ve = np.cbrt(rga_ve_d * rge_ve_d * rgr_ve_d)
values_diff_g_nn = np.cbrt(rga_nn_d * rge_nn_d * rgr_nn_d)

plot_metric_distribution_diff(
    metric_values=values_diff_g_lr,
    print_label="Difference Geometric Mean (1/3) Logistic Regression",
    xlabel="Difference Geometric Mean (1/3)",
    title="Histogram of Difference Geometric Mean (1/3) Values (Logistic Regression)",
    bar_label="Logistic Regression"
)

plot_metric_distribution_diff(
    metric_values=values_diff_g_rf,
    print_label="Difference Geometric Mean (1/3) Random Forest",
    xlabel="Difference Geometric Mean (1/3)",
    title="Histogram of Difference Geometric Mean (1/3) Values (Random Forest)",
    bar_label="Random Forest"
)

plot_metric_distribution_diff(
    metric_values=values_diff_g_xgb,
    print_label="Difference Geometric Mean (1/3) XGBoosting",
    xlabel="Difference Geometric Mean (1/3)",
    title="Histogram of Difference Geometric Mean (1/3) Values (XGBoosting)",
    bar_label="XGBoosting"
)

plot_metric_distribution_diff(
    metric_values=values_diff_g_se,
    print_label="Difference Geometric Mean (1/3) Stacked Ensemble",
    xlabel="Difference Geometric Mean (1/3)",
    title="Histogram of Difference Geometric Mean (1/3) Values (Stacked Ensemble)",
    bar_label="Stacked Ensemble"
)

plot_metric_distribution_diff(
    metric_values=values_diff_g_ve,
    print_label="Difference Geometric Mean (1/3) Voting Ensemble",
    xlabel="Difference Geometric Mean (1/3)",
    title="Histogram of Difference Geometric Mean (1/3) Values (Voting Ensemble)",
    bar_label="Voting Ensemble"
)

plot_metric_distribution_diff(
    metric_values=values_diff_g_nn,
    print_label="Difference Geometric Mean (1/3) Neural Network",
    xlabel="Difference Geometric Mean (1/3)",
    title="Histogram of Difference Geometric Mean (1/3) Values (Neural Network)",
    bar_label="Neural Network"
)

plt.show()

# Differences Plots Quadratic Mean (RMS)
values_diff_rms_lr = np.sqrt((rga_lr_d ** 2 + rge_lr_d ** 2 + rgr_lr_d ** 2) / 3)
values_diff_rms_rf = np.sqrt((rga_rf_d ** 2 + rge_rf_d ** 2 + rgr_rf_d ** 2) / 3)
values_diff_rms_xgb = np.sqrt((rga_xgb_d ** 2 + rge_xgb_d ** 2 + rgr_xgb_d ** 2) / 3)
values_diff_rms_se = np.sqrt((rga_se_d ** 2 + rge_se_d ** 2 + rgr_se_d ** 2) / 3)
values_diff_rms_ve = np.sqrt((rga_ve_d ** 2 + rge_ve_d ** 2 + rgr_ve_d ** 2) / 3)
values_diff_rms_nn = np.sqrt((rga_nn_d ** 2 + rge_nn_d ** 2 + rgr_nn_d ** 2) / 3)

plot_metric_distribution_diff(
    metric_values=values_diff_rms_lr,
    print_label="Difference Quadratic Mean (RMS) Logistic Regression",
    xlabel="Difference Quadratic Mean (RMS)",
    title="Histogram of Difference Quadratic Mean (RMS) Values (Logistic Regression)",
    bar_label="Logistic Regression"
)

plot_metric_distribution_diff(
    metric_values=values_diff_rms_rf,
    print_label="Difference Quadratic Mean (RMS) Random Forest",
    xlabel="Difference Quadratic Mean (RMS)",
    title="Histogram of Difference Quadratic Mean (RMS) Values (Random Forest)",
    bar_label="Random Forest"
)

plot_metric_distribution_diff(
    metric_values=values_diff_rms_xgb,
    print_label="Difference Quadratic Mean (RMS) XGBoosting",
    xlabel="Difference Quadratic Mean (RMS)",
    title="Histogram of Difference Quadratic Mean (RMS) Values (XGBoosting)",
    bar_label="XGBoosting"
)

plot_metric_distribution_diff(
    metric_values=values_diff_rms_se,
    print_label="Difference Quadratic Mean (RMS) Stacked Ensemble",
    xlabel="Difference Quadratic Mean (RMS)",
    title="Histogram of Difference Quadratic Mean (RMS) Values (Stacked Ensemble)",
    bar_label="Stacked Ensemble"
)

plot_metric_distribution_diff(
    metric_values=values_diff_rms_ve,
    print_label="Difference Quadratic Mean (RMS) Voting Ensemble",
    xlabel="Difference Quadratic Mean (RMS)",
    title="Histogram of Difference Quadratic Mean (RMS) Values (Voting Ensemble)",
    bar_label="Voting Ensemble"
)

plot_metric_distribution_diff(
    metric_values=values_diff_rms_nn,
    print_label="Difference Quadratic Mean (RMS) Neural Network",
    xlabel="Difference Quadratic Mean (RMS)",
    title="Histogram of Difference Quadratic Mean (RMS) Values (Neural Network)",
    bar_label="Neural Network"
)

plt.show()