import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from safeai_files.check_compliance import safeai_values
from safeai_files.utils import plot_mean_histogram, plot_model_curves, plot_diff_mean_histogram

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

# Column for Classification task
data["doubling_salary"] = np.where(data["salary"]/data["startsal"] > 2,1,0)
print(data["doubling_salary"].value_counts())
data.drop(["salary", "startsal"], axis=1, inplace=True)
print(data.head())

# Split to train and test
x = data.drop("doubling_salary", axis=1)
y_class = data["doubling_salary"]

x_train, x_test, y_train_cl, y_test_cl = train_test_split(x, y_class, test_size=0.3, random_state=1)

# Classification problem
# Logistic Regression
log_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=123)
log_model.fit(x_train, y_train_cl)
y_prob_lr = log_model.predict_proba(x_test)[:, 1]
results_log = safeai_values(x_train, x_test, y_test_cl, y_prob_lr, log_model, "Employee Classification", "plots")
print(results_log)

# Random Forest
rf_model = RandomForestClassifier(random_state=123)
rf_model.fit(x_train, y_train_cl)
y_prob_rf = rf_model.predict_proba(x_test)[:, 1]
results_rf = safeai_values(x_train, x_test, y_test_cl, y_prob_rf, rf_model, "Employee Classification", "plots")
print(results_rf)

# XGBoosting
xgb_model = xgb.XGBClassifier(random_state=123)
xgb_model.fit(x_train, y_train_cl)
y_prob_xgb = xgb_model.predict_proba(x_test)[:, 1]
results_xgb = safeai_values(x_train, x_test, y_test_cl, y_prob_xgb, xgb_model, "Employee Classification", "plots")
print(results_xgb)

# Stacked Ensemble Model
stacking_clf = StackingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], final_estimator=log_model)
stacking_clf.fit(x_train, y_train_cl)
y_prob_se = stacking_clf.predict_proba(x_test)[:, 1]
results_se = safeai_values(x_train, x_test, y_test_cl, y_prob_se, stacking_clf, "Employee Classification", "plots")
print(results_se)

# Voting Ensemble Model
voting_clf = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')
voting_clf.fit(x_train, y_train_cl)
y_prob_ve = voting_clf.predict_proba(x_test)[:, 1]
results_ve = safeai_values(x_train, x_test, y_test_cl, y_prob_ve, voting_clf, "Employee Classification", "plots")
print(results_ve)

# Random Model
random_model = DummyClassifier(random_state=123)
random_model.fit(x_train, y_train_cl)
y_prob_r = random_model.predict_proba(x_test)[:, 1]
results_r = safeai_values(x_train, x_test, y_test_cl, y_prob_r, random_model, "Employee Classification", "plots")
print(results_r)

# Extract values
x_lr = results_log["x_final"]
y_lr = results_log["y_final"]
z_lr = results_log["z_final"]

x_rf = results_rf["x_final"]
y_rf = results_rf["y_final"]
z_rf = results_rf["z_final"]

x_xgb = results_xgb["x_final"]
y_xgb = results_xgb["y_final"]
z_xgb = results_xgb["z_final"]

x_se = results_se["x_final"]
y_se = results_se["y_final"]
z_se = results_se["z_final"]

x_ve = results_ve["x_final"]
y_ve = results_ve["y_final"]
z_ve = results_ve["z_final"]

x_r = results_r["x_final"]
y_r = results_r["y_final"]
z_r = results_r["z_final"]

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

# Compliance Curves (Classification)
# Logistic Regression
x_step = np.linspace(0, 1, len(y_r))
plot_model_curves(x_step, [x_lr, y_lr, z_lr], model_name="Logistic Regression",
                  title="Logistic Regression Curves (Classification)")

# Random Forest
plot_model_curves(x_step, [x_rf, y_rf, z_rf], model_name="Random Forest",
                  title="Random Forest Curves (Classification)")

# XGBoosting
plot_model_curves(x_step,[x_xgb, y_xgb, z_xgb], model_name="XGBoosting",
                  title="XGBoosting Curves (Classification)")

# Stacked Ensemble Model
plot_model_curves(x_step,[x_se, y_se, z_se], model_name="Stacked Ensemble Model",
                  title="Stacked Ensemble Model Curves (Classification)")

# Voting Ensemble Model
plot_model_curves(x_step,[x_ve, y_ve, z_ve], model_name="Voting Ensemble Model",
                  title="Voting Ensemble Model Curves (Classification)")

# Random Model
plot_model_curves(x_step,[x_r, y_r, z_r], model_name="Random Model",
                  title="Random Model Curves (Classification)")

# Difference LR and Random
plot_model_curves(x_step,[x_lr_r, y_lr_r, z_lr_r], model_name="Random", prefix="Difference",
                  title="LR and Random Curves Difference (Classification)")

# Difference RF and Random
plot_model_curves(x_step,[x_rf_r, y_rf_r, z_rf_r], model_name="Random", prefix="Difference",
                  title="RF and Random Curves Difference (Classification)")

# Difference XGB and Random
plot_model_curves(x_step,[x_xgb_r, y_xgb_r, z_xgb_r], model_name="Random", prefix="Difference",
                  title="XGB and Random Curves Difference (Classification)")

# Difference SE and Random
plot_model_curves(x_step,[x_se_r, y_se_r, z_se_r], model_name="Random", prefix="Difference",
                  title="SE and Random Curves Difference (Classification)")

# Difference VE and Random
plot_model_curves(x_step,[x_ve_r, y_ve_r, z_ve_r], model_name="Random", prefix="Difference",
                  title="VE and Random Curves Difference (Classification)")

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

# Means
models = [
    ((rga_lr,  rge_lr,  rgr_lr),  "Logistic Regression", "Logistic Regression"),
    ((rga_rf,  rge_rf,  rgr_rf),  "Random Forest", "Random Forest Model"),
    ((rga_xgb, rge_xgb, rgr_xgb), "XGBoosting", "XGB Model"),
    ((rga_se,  rge_se,  rgr_se),  "Stacked Ensemble", "Stacked Ensemble Model"),
    ((rga_ve,  rge_ve,  rgr_ve),  "Voting Ensemble", "Voting Ensemble Model"),
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

models_diff = [
    ((rga_lr_d,  rge_lr_d,  rgr_lr_d),  "Logistic Regression", "Logistic Regression"),
    ((rga_rf_d,  rge_rf_d,  rgr_rf_d),  "Random Forest", "Random Forest Model"),
    ((rga_xgb_d, rge_xgb_d, rgr_xgb_d), "XGBoosting", "XGB Model"),
    ((rga_se_d,  rge_se_d,  rgr_se_d),  "Stacked Ensemble", "Stacked Ensemble Model"),
    ((rga_ve_d,  rge_ve_d,  rgr_ve_d),  "Voting Ensemble", "Voting Ensemble Model"),
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

# Hypervolume approach
def hypervolume(x1, y2, z3):
    v1 = np.array(x1)
    v2 = np.array(y2)
    v3 = np.array(z3)

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

# Hypervolume R
volume_r = hypervolume(rgas_random, rges_random, rgrs_random)
print(f"Hypervolume Random: {volume_r:.3f}")