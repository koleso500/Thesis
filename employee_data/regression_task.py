import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from safeai_files.check_compliance import compliance_curves
from safeai_files.utils import plot_model_curves, plot_metric_distribution, plot_metric_distribution_diff

# Data loading and basic information
data = pd.read_excel("employee.xlsx")
print(data.shape)
print(data.columns)
print("This dataset has {} rows and {} columns".format(data.shape[0], data.shape[1]))
types = data.dtypes
print(types)

# Check NaNs
print(data.isna().sum())  # Number of NaNs per column
print(data.isna().sum().sum()) # Total number of NaNs in the entire data

# Change to int
print(data.head())
print(data["gender"].value_counts())
print(data["minority"].value_counts())

data["gender"] = np.where(data["gender"]=="m", 0, 1)
data["minority"] = np.where(data["minority"]=="no_min", 0, 1)

# Column for Regression task
data["salary_growth"] = data["salary"]-data["startsal"]
data.drop(["salary", "startsal"], axis=1, inplace=True)
print(data.head())

# Split to train and test
x = data.drop("salary_growth", axis=1)
y_reg = data["salary_growth"]

x_train, x_test, y_train_rg, y_test_rg = train_test_split(x, y_reg, test_size=0.3, random_state=1)

# Regression problem
# Linear Regression
reg_model = LinearRegression()
reg_model.fit(x_train, y_train_rg)
y_prob_reg = reg_model.predict(x_test)
results_reg = compliance_curves(x_train, x_test, y_test_rg, y_prob_reg, reg_model)
print(results_reg)

# Random Forest
rf_rg = RandomForestRegressor(random_state=123)
rf_rg.fit(x_train, y_train_rg)
y_prob_rf_rg = rf_rg.predict(x_test)
results_rf_rg = compliance_curves(x_train, x_test, y_test_rg, y_prob_rf_rg, rf_rg)
print(results_rf_rg)

# XGBoosting
xgb_rg = xgb.XGBRegressor(random_state=123)
xgb_rg.fit(x_train, y_train_rg)
y_prob_xgb_rg = xgb_rg.predict(x_test)
results_xgb_rg = compliance_curves(x_train, x_test, y_test_rg, y_prob_xgb_rg, xgb_rg)
print(results_xgb_rg)

# Stacked Ensemble Model
stacking_rg = StackingRegressor(estimators=[('rf', rf_rg), ('xgb', xgb_rg)], final_estimator=reg_model)
stacking_rg.fit(x_train, y_train_rg)
y_prob_se_rg = stacking_rg.predict(x_test)
results_se_rg = compliance_curves(x_train, x_test, y_test_rg, y_prob_se_rg, stacking_rg)
print(results_se_rg)

# Voting Ensemble Model
voting_rg = VotingRegressor(estimators=[('rf', rf_rg), ('xgb', xgb_rg)])
voting_rg.fit(x_train, y_train_rg)
y_prob_ve_rg = voting_rg.predict(x_test)
results_ve_rg = compliance_curves(x_train, x_test, y_test_rg, y_prob_ve_rg, voting_rg)
print(results_ve_rg)

# Random Model
random_rg = DummyRegressor()
random_rg.fit(x_train, y_train_rg)
y_prob_r_rg = random_rg.predict(x_test)
results_r_rg = compliance_curves(x_train, x_test, y_test_rg, y_prob_r_rg, random_rg)
print(results_r_rg)

# Extract values
x_lr = results_reg["x_final"]
y_lr = results_reg["y_final"]
z_lr = results_reg["z_final"]

x_rf = results_rf_rg["x_final"]
y_rf = results_rf_rg["y_final"]
z_rf = results_rf_rg["z_final"]

x_xgb = results_xgb_rg["x_final"]
y_xgb = results_xgb_rg["y_final"]
z_xgb = results_xgb_rg["z_final"]

x_se = results_se_rg["x_final"]
y_se = results_se_rg["y_final"]
z_se = results_se_rg["z_final"]

x_ve = results_ve_rg["x_final"]
y_ve = results_ve_rg["y_final"]
z_ve = results_ve_rg["z_final"]

x_r = results_r_rg["x_final"]
y_r = results_r_rg["y_final"]
z_r = results_r_rg["z_final"]

# Differences
x_lr_r = (np.array(x_lr) - np.array(x_r)).tolist()
y_lr_r = (np.array(y_lr) - np.array(y_r)).tolist()
z_lr_r = (np.array(z_lr) - np.array(z_r)).tolist()

x_rf_r = (np.array(x_rf) - np.array(x_r)).tolist()
y_rf_r = (np.array(y_rf) - np.array(y_r)).tolist()
z_rf_r = (np.array(z_rf) - np.array(z_r)).tolist()

x_xgb_r = (np.array(x_xgb) - np.array(x_r)).tolist()
y_xgb_r = (np.array(y_xgb) - np.array(y_r)).tolist()
z_xgb_r = (np.array(z_xgb) - np.array(z_r)).tolist()

x_se_r = (np.array(x_se) - np.array(x_r)).tolist()
y_se_r = (np.array(y_se) - np.array(y_r)).tolist()
z_se_r = (np.array(z_se) - np.array(z_r)).tolist()

x_ve_r = (np.array(x_ve) - np.array(x_r)).tolist()
y_ve_r = (np.array(y_ve) - np.array(y_r)).tolist()
z_ve_r = (np.array(z_ve) - np.array(z_r)).tolist()

# Compliance Curves (Regression)
# Linear Regression
x_step = np.linspace(0, 1, len(y_r))
plot_model_curves(x_step,[x_lr, y_lr, z_lr], model_name="Linear Regression",
                  title="Linear Regression Curves (Regression)")

# Random Forest
plot_model_curves(x_step,[x_rf, y_rf, z_rf], model_name="Random Forest",
                  title="Random Forest Curves (Regression)")

# XGBoosting
plot_model_curves(x_step,[x_xgb, y_xgb, z_xgb], model_name="XGBoosting",
                  title="XGBoosting Curves (Regression)")

# Stacked Ensemble Model
plot_model_curves(x_step,[x_se, y_se, z_se], model_name="Stacked Ensemble Model",
                  title="Stacked Ensemble Model Curves (Regression)")

# Voting Ensemble Model
plot_model_curves(x_step,[x_ve, y_ve, z_ve], model_name="Voting Ensemble Model",
                  title="Voting Ensemble Model Curves (Regression)")

# Random Model
plot_model_curves(x_step,[x_r, y_r, z_r], model_name="Random Model",
                  title="Random Model Curves (Regression)")

# Difference LR and Random
plot_model_curves(x_step,[x_lr_r, y_lr_r, z_lr_r], model_name="Random", prefix="Difference",
                  title="LR and Random Curves Difference (Regression)")

# Difference RF and Random
plot_model_curves(x_step,[x_rf_r, y_rf_r, z_rf_r], model_name="Random", prefix="Difference",
                  title="RF and Random Curves Difference (Regression)")

# Difference XGB and Random
plot_model_curves(x_step,[x_xgb_r, y_xgb_r, z_xgb_r], model_name="Random", prefix="Difference",
                  title="XGB and Random Curves Difference (Regression)")

# Difference SE and Random
plot_model_curves(x_step,[x_se_r, y_se_r, z_se_r], model_name="Random", prefix="Difference",
                  title="SE and Random Curves Difference (Regression)")

# Difference VE and Random
plot_model_curves(x_step,[x_ve_r, y_ve_r, z_ve_r], model_name="Random", prefix="Difference",
                  title="VE and Random Curves Difference (Regression)")

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

rgas_se = np.array(x_se)
rges_se = np.array(y_se)
rgrs_se = np.array(z_se)

rgas_ve = np.array(x_ve)
rges_ve = np.array(y_ve)
rgrs_ve = np.array(z_ve)

rgas_random = np.array(x_r)
rges_random = np.array(y_r)
rgrs_random = np.array(z_r)

# Scalar fields, matrix of initial values
rga_lr, rge_lr, rgr_lr = np.meshgrid(rgas_lr, rges_lr, rgrs_lr, indexing='ij')
rga_rf, rge_rf, rgr_rf = np.meshgrid(rgas_rf, rges_rf, rgrs_rf, indexing='ij')
rga_xgb, rge_xgb, rgr_xgb = np.meshgrid(rgas_xgb, rges_xgb, rgrs_xgb, indexing='ij')
rga_se, rge_se, rgr_se = np.meshgrid(rgas_se, rges_se, rgrs_se, indexing='ij')
rga_ve, rge_ve, rgr_ve = np.meshgrid(rgas_ve, rges_ve, rgrs_ve, indexing='ij')
rga_r, rge_r, rgr_r = np.meshgrid(rgas_random, rges_random, rgrs_random, indexing='ij')

# Histogram of the arithmetic mean metric LR
values_a_lr = (rga_lr + rge_lr + rgr_lr) / 3
plot_metric_distribution(
    metric_values=values_a_lr,
    print_label="Mean volume Arithmetic Linear Regression",
    xlabel="Normalized Arithmetic Mean",
    title="Histogram of Normalized Arithmetic Mean Values (Linear Regression)",
    bar_label="Linear Regression"
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
    print_label="Mean volume Geometric Linear Regression",
    xlabel="Normalized Geometric Mean (1/3)",
    title="Histogram of Normalized Geometric Mean (1/3) Values (Linear Regression)",
    bar_label="Linear Regression"
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
    print_label="Mean volume Quadratic Mean (RMS) Linear Regression",
    xlabel="Normalized Quadratic Mean (RMS)",
    title="Histogram of Normalized Quadratic Mean (RMS) Values (Linear Regression)",
    bar_label="Linear Regression"
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

rga_d_se = np.array(x_se_r)
rge_d_se = np.array(y_se_r)
rgr_d_se = np.array(z_se_r)

rga_d_ve = np.array(x_ve_r)
rge_d_ve = np.array(y_ve_r)
rgr_d_ve = np.array(z_ve_r)

rga_lr_d, rge_lr_d, rgr_lr_d = np.meshgrid(rga_d_lr, rge_d_lr, rgr_d_lr, indexing='ij')
rga_rf_d, rge_rf_d, rgr_rf_d = np.meshgrid(rga_d_rf, rge_d_rf, rgr_d_rf, indexing='ij')
rga_xgb_d, rge_xgb_d, rgr_xgb_d = np.meshgrid(rga_d_xgb, rge_d_xgb, rgr_d_xgb, indexing='ij')
rga_se_d, rge_se_d, rgr_se_d = np.meshgrid(rga_d_se, rge_d_se, rgr_d_se, indexing='ij')
rga_ve_d, rge_ve_d, rgr_ve_d = np.meshgrid(rga_d_ve, rge_d_ve, rgr_d_ve, indexing='ij')

values_diff_lr = (rga_lr_d + rge_lr_d + rgr_lr_d) / 3
values_diff_rf = (rga_rf_d + rge_rf_d + rgr_rf_d) / 3
values_diff_xgb = (rga_xgb_d + rge_xgb_d + rgr_xgb_d) / 3
values_diff_se = (rga_se_d + rge_se_d + rgr_se_d) / 3
values_diff_ve = (rga_ve_d + rge_ve_d + rgr_ve_d) / 3


plot_metric_distribution_diff(
    metric_values=values_diff_lr,
    print_label="Difference Arithmetic Linear Regression",
    xlabel="Difference Arithmetic Mean",
    title="Histogram of Difference Arithmetic Mean Values (Linear Regression)",
    bar_label="Linear Regression"
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

plt.show()

# Differences Plots Geometric Mean (1/3)
values_diff_g_lr = np.cbrt(rga_lr_d * rge_lr_d * rgr_lr_d)
values_diff_g_rf = np.cbrt(rga_rf_d * rge_rf_d * rgr_rf_d)
values_diff_g_xgb = np.cbrt(rga_xgb_d * rge_xgb_d * rgr_xgb_d)
values_diff_g_se = np.cbrt(rga_se_d * rge_se_d * rgr_se_d)
values_diff_g_ve = np.cbrt(rga_ve_d * rge_ve_d * rgr_ve_d)

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

plt.show()

# Differences Plots Quadratic Mean (RMS)
values_diff_rms_lr = np.sqrt((rga_lr_d ** 2 + rge_lr_d ** 2 + rgr_lr_d ** 2) / 3)
values_diff_rms_rf = np.sqrt((rga_rf_d ** 2 + rge_rf_d ** 2 + rgr_rf_d ** 2) / 3)
values_diff_rms_xgb = np.sqrt((rga_xgb_d ** 2 + rge_xgb_d ** 2 + rgr_xgb_d ** 2) / 3)
values_diff_rms_se = np.sqrt((rga_se_d ** 2 + rge_se_d ** 2 + rgr_se_d ** 2) / 3)
values_diff_rms_ve = np.sqrt((rga_ve_d ** 2 + rge_ve_d ** 2 + rgr_ve_d ** 2) / 3)

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

plt.show()