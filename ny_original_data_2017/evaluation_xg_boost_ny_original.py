import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_curve
import xgboost as xgb

from safeai_files.check_explainability import compute_rge_values
from safeai_files.check_fairness import compute_rga_parity
from safeai_files.check_robustness import compute_rgr_values, rgr_all, rgr_single
from safeai_files.core import rga

# Load best model and variables
best_model = joblib.load("../saved_models/best_xgb_model_ny_original.joblib")
x_train = pd.read_csv("../saved_data/x_train_xgb_ny_original.csv")
x_test = pd.read_csv("../saved_data/x_test_xgb_ny_original.csv")
y_train = pd.read_csv("../saved_data/y_train_xgb_ny_original.csv")
y_test = pd.read_csv("../saved_data/y_test_xgb_ny_original.csv")
y_test = y_test.values.tolist()

# Make predictions
y_pred = best_model.predict(x_test)
y_prob = best_model.predict_proba(x_test)[:, 1]

# AUC (for future use)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:\n", roc_auc)

# Partial AUC
fpr_threshold = 0.3
valid_indices = np.where(fpr < fpr_threshold)[0]
partial_auc = auc(fpr[valid_indices], tpr[valid_indices]) / fpr_threshold
print(f"Partial AUC (FPR < {fpr_threshold}): {partial_auc:.4f}")

# ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()

# DataFrame with thresholds, FPR, TPR, and F1 Score
roc_data = pd.DataFrame({
    'threshold': thresholds,
    'fpr': fpr,
    'tpr': tpr,
    'f1_score': [f1_score(y_test, np.where(y_prob >= t, 1, 0)) for t in thresholds]
})

# Best F1 score and corresponding threshold
best_f1 = roc_data.loc[roc_data['f1_score'].idxmax()]
best_threshold = best_f1["threshold"]
print(f"Best F1 Score: {best_f1["f1_score"]:.4f} at Threshold: {best_threshold:.4f}")

# Predictions at best threshold
y_pred_best = np.where(y_prob >= best_threshold, 1, 0)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))

# Feature importance plot
xgb.plot_importance(best_model)
plt.show()

# Integrating safeai
# Accuracy
rga_class = rga(y_test, y_prob)
print(f"RGA value is equal to {rga_class}")

# Explainability
# RGE AUC
# Get values of RGE for each variable
explain = x_train.columns.tolist()
single_rge = compute_rge_values(x_train, x_test, y_prob, best_model, explain)
print(single_rge)
single_rge = single_rge.index.tolist()

# Get all group values of RGE
step_rges = []
for k in range(0, len(single_rge) + 1):
    top_k_vars = single_rge[:k]
    rge_k = compute_rge_values(x_train, x_test, y_prob, best_model, top_k_vars, group=True)
    step_rges.append(rge_k.iloc[0, 0])

# Normalize
x_rge = np.linspace(0, 1, len(step_rges))
y_rge = np.array(step_rges)
y_rge /= y_rge.max()

# Compute RGE AUC
rge_score = auc(x_rge, y_rge)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x_rge, y_rge, marker='o')
plt.xlabel("Fraction of Top Variables Removed")
plt.ylabel("RGE")
plt.title("XGBoost RGE Curve (New York Original)")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"RGE AUC: {rge_score:.4f}")

# Fairness
gender = compute_rga_parity(x_train, x_test, y_test, y_prob, best_model, "applicant_sex")
print("Gender:\n", gender)
race = compute_rga_parity(x_train, x_test, y_test, y_prob, best_model, "applicant_race_1")
print("Race:\n", race)

# Robustness
print(compute_rgr_values(x_test, y_prob, best_model, list(x_test.columns), 0.3))
print(rgr_single(x_test, y_prob, best_model, "loan_purpose", 0.2))

# RGR AUC
thresholds = np.arange(0, 0.51, 0.01)
results = [rgr_all(x_test, y_prob, best_model, t) for t in thresholds]
normalized_thresholds = thresholds / 0.5

plt.figure(figsize=(8, 5))
plt.plot(normalized_thresholds, results, linestyle='-')
plt.title('XGBoost RGR Curve (New York Original)')
plt.xlabel('Normalized Perturbation')
plt.ylabel('RGR')
plt.grid(True)
plt.tight_layout()
plt.show()
rgr_auc = auc(normalized_thresholds, results)
print(f"RGR AUC: {rgr_auc:.4f}")