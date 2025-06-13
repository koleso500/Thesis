import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from safeai_files.utils import plot_mean_histogram, plot_model_curves, plot_metric_distribution, plot_diff_mean_histogram

# Load values
file_path_lr = os.path.join("../saved_data", "final_results_lr_ny_article.json")
file_path_rf = os.path.join("../saved_data", "final_results_rf_ny_article.json")
file_path_xgb = os.path.join("../saved_data", "final_results_xgb_ny_article.json")
file_path_stacked = os.path.join("../saved_data", "final_results_stacked_ny_article.json")
file_path_voting = os.path.join("../saved_data", "final_results_voting_ny_article.json")
file_path_neural = os.path.join("../saved_data", "final_results_neural_ny_article.json")
file_path_random = os.path.join("../saved_data", "final_results_random_ny_article.json")

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
plot_model_curves(x_rga,[x_lr, y_lr, z_lr], model_name="LR", title="Logistic Regression Curves (New York Article)")

# All curves for RF
plot_model_curves(x_rga,[x_rf, y_rf, z_rf], model_name="RF", title="Random Forest Curves (New York Article)")

# All curves for XGB
plot_model_curves(x_rga,[x_xgb, y_xgb, z_xgb], model_name="XGB", title="XGBoosting Curves (New York Article)")

# All curves for SE
plot_model_curves(x_rga,[x_stacked, y_stacked, z_stacked], model_name="SE", title="Stacked Ensemble Curves (New York Article)")

# All curves for VE
plot_model_curves(x_rga,[x_voting, y_voting, z_voting], model_name="VE", title="Voting Ensemble Curves (New York Article)")

# All curves for NN
plot_model_curves(x_rga,[x_neural, y_neural, z_neural], model_name="NN", title="Neural Network Curves (New York Article)")

# All curves for Random
plot_model_curves(x_rga,[x_random, y_random, z_random], model_name="Random",title="Random Classifier Curves (New York Article)")

# All curves for difference LR and Random
plot_model_curves(x_rga,[x_lr_r, y_lr_r, z_lr_r], model_name="Random", prefix="Difference",
                  title="LR and Random Curves Difference (New York Article)")

# All curves for difference RF and Random
plot_model_curves(x_rga,[x_rf_r, y_rf_r, z_rf_r], model_name="Random", prefix="Difference",
                  title="RF and Random Curves Difference (New York Article)")

# All curves for difference XGB and Random
plot_model_curves(x_rga,[x_xgb_r, y_xgb_r, z_xgb_r], model_name="Random", prefix="Difference",
                  title="XGB and Random Curves Difference (New York Article)")

# All curves for difference SE and Random
plot_model_curves(x_rga,[x_stacked_r, y_stacked_r, z_stacked_r], model_name="Random", prefix="Difference",
                  title="SE and Random Curves Difference (New York Article)")

# All curves for difference VE and Random
plot_model_curves(x_rga,[x_voting_r, y_voting_r, z_voting_r], model_name="Random", prefix="Difference",
                  title="VE and Random Curves Difference (New York Article)")

# All curves for difference NN and Random
plot_model_curves(x_rga,[x_neural_r, y_neural_r, z_neural_r], model_name="Random", prefix="Difference",
                  title="NN and Random Curves Difference (New York Article)")

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

# TOPSIS approach
best_x_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
worst_x_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

best_y_list = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
worst_y_list = [1.0, 0.9736842105263158, 0.9473684210526316, 0.9210526315789473, 0.8947368421052632, 0.868421052631579, 0.8421052631578947, 0.8157894736842105, 0.7894736842105263, 0.7631578947368421, 0.736842105263158, 0.7105263157894737, 0.6842105263157895, 0.6578947368421053, 0.631578947368421, 0.6052631578947368, 0.5789473684210527, 0.5526315789473685, 0.5263157894736843, 0.5 ]

best_z_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
worst_z_list = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

x_plus = np.mean(best_x_list)
x_minus = np.mean(worst_x_list)

y_plus = np.mean(best_y_list)
y_minus = np.mean(worst_y_list)

z_plus = np.mean(best_z_list)
z_minus = np.mean(worst_z_list)

mean_x_lr = np.mean(rgas_lr)
mean_y_lr = np.mean(rges_lr)
mean_z_lr = np.mean(rgrs_lr)

mean_x_rf = np.mean(rgas_rf)
mean_y_rf = np.mean(rges_rf)
mean_z_rf = np.mean(rgrs_rf)

mean_x_xgb = np.mean(rgas_xgb)
mean_y_xgb = np.mean(rges_xgb)
mean_z_xgb = np.mean(rgrs_xgb)

mean_x_se = np.mean(rgas_se)
mean_y_se = np.mean(rges_se)
mean_z_se = np.mean(rgrs_se)

mean_x_ve = np.mean(rgas_ve)
mean_y_ve = np.mean(rges_ve)
mean_z_ve = np.mean(rgrs_ve)

mean_x_nn = np.mean(rgas_nn)
mean_y_nn = np.mean(rges_nn)
mean_z_nn = np.mean(rgrs_nn)

mean_x_r = np.mean(rgas_random)
mean_y_r = np.mean(rges_random)
mean_z_r = np.mean(rgrs_random)

means = {
    "Logistic":       (mean_x_lr, mean_y_lr, mean_z_lr),
    "RandomForest":   (mean_x_rf, mean_y_rf, mean_z_rf),
    "XGBoost":        (mean_x_xgb, mean_y_xgb, mean_z_xgb),
    "StackedEnsemble":(mean_x_se, mean_y_se, mean_z_se),
    "VotingEnsemble": (mean_x_ve,  mean_y_ve,  mean_z_ve),
    "NeuralNetwork":  (mean_x_nn,  mean_y_nn,  mean_z_nn),
    "RandomBaseline": (mean_x_r,  mean_y_r,  mean_z_r),
}

df = pd.DataFrame.from_dict(
    means,
    orient="index",
    columns=["mean_x", "mean_y", "mean_z"]
)

for col in ["mean_x", "mean_y", "mean_z"]:
    vec = df[col].values.astype(float)
    norm = np.sqrt((vec**2).sum())
    df["r_" + col] = vec / norm

weights = np.array([0.9, 0.05, 0.05])
df["v_mean_x"] = df["r_mean_x"] * weights[0]
df["v_mean_y"] = df["r_mean_y"] * weights[1]
df["v_mean_z"] = df["r_mean_z"] * weights[2]

norm_x = np.sqrt((df["mean_x"].values ** 2).sum())
norm_y = np.sqrt((df["mean_y"].values ** 2).sum())
norm_z = np.sqrt((df["mean_z"].values ** 2).sum())

r_x_plus = x_plus / norm_x
r_x_minus = x_minus / norm_x

r_y_plus = y_plus / norm_y
r_y_minus = y_minus / norm_y

r_z_plus = z_plus / norm_z
r_z_minus = z_minus / norm_z

v_x_plus  = r_x_plus  * weights[0]
v_x_minus = r_x_minus * weights[0]

v_y_plus  = r_y_plus  * weights[1]
v_y_minus = r_y_minus * weights[1]

v_z_plus  = r_z_plus  * weights[2]
v_z_minus = r_z_minus * weights[2]

df["S_plus"] = np.sqrt(
    (df["v_mean_x"] - v_x_plus)**2 +
    (df["v_mean_y"] - v_y_plus)**2 +
    (df["v_mean_z"] - v_z_plus)**2
)

df["S_minus"] = np.sqrt(
    (df["v_mean_x"] - v_x_minus)**2 +
    (df["v_mean_y"] - v_y_minus)**2 +
    (df["v_mean_z"] - v_z_minus)**2
)

df["C"] = df["S_minus"] / (df["S_plus"] + df["S_minus"])
df["Rank"] = df["C"].rank(ascending=False)
df_sorted = df.sort_values("C", ascending=False)
print(df_sorted[["C", "Rank"]])

# Means approach
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