import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from safeai_files.check_explainability import compute_rge_values
from safeai_files.check_robustness import rgr_all
from safeai_files.core import partial_rga_with_curves, rga

# Data Separation
data_lending_ny_clean = pd.read_csv("../saved_data/data_lending_clean_ny_original.csv")
predictors = data_lending_ny_clean.drop(columns=['action_taken'])
y = data_lending_ny_clean['action_taken']

# Splitting into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(predictors, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Load best parameters for models
with open(os.path.join("../saved_data", "best_rf_params_ny_original.json"), "r", encoding="utf-8") as file_rf:
    best_rf_params = json.load(file_rf)

with open(os.path.join("../saved_data", "best_xgb_params_ny_original.json"), "r", encoding="utf-8") as file_xgb:
    best_xgb_params = json.load(file_xgb)

# Models
rf_clf = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
xgb_clf = XGBClassifier(**best_xgb_params, objective='binary:logistic', eval_metric='logloss')

# Voting Classifier with Soft Voting
voting_clf = VotingClassifier(estimators=[('rf', rf_clf), ('xgb', xgb_clf)], voting='soft')

# Train
voting_clf.fit(x_train, y_train)

# Make predictions
y_pred = voting_clf.predict(x_test)
y_prob = voting_clf.predict_proba(x_test)[:, 1]

#AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:\n", roc_auc)

# Partial AUC
fpr_threshold = 0.3
valid_indices = np.where(fpr < fpr_threshold)[0]
partial_auc = auc(fpr[valid_indices], tpr[valid_indices]) / fpr_threshold
print(f"Partial AUC (FPR < {fpr_threshold}): {partial_auc:.4f}")

#ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.savefig("plots/VEM_ROC_Curve_ny_original.png", dpi=300)
plt.close()

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

# Integrating safeai
# Accuracy
rga_class = rga(y_test, y_prob)
print(f"RGA value is equal to {rga_class}")

# Explainability
# RGE AUC
explain = x_train.columns.tolist()
remaining_vars = explain.copy()
removed_vars = []
step_rges = []

for k in range(0, len(explain) + 1):
    if k == 0:
        step_rges.append(0.0)
        continue

    candidate_rges = []
    for var in remaining_vars:
        current_vars = removed_vars + [var]
        rge_k = compute_rge_values(x_train, x_test, y_prob, voting_clf, current_vars, group=True)
        candidate_rges.append((var, rge_k.iloc[0, 0]))

    best_var, best_rge = max(candidate_rges, key=lambda x: x[1])
    removed_vars.append(best_var)
    remaining_vars.remove(best_var)
    step_rges.append(best_rge)

# Compute AURGE
x_rge = np.linspace(0, 1, len(step_rges))
y_rge = np.array(step_rges)
rge_auc = auc(x_rge, y_rge)
print(f"AURGE: {rge_auc:.4f}")

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x_rge, y_rge, marker='o', label=f"RGE Curve (AURGE = {rge_auc:.4f})")
random_baseline = float(y_rge[-1])
plt.axhline(random_baseline, color='red', linestyle='--', label="Random Classifier (RGE = 0.5)")
plt.xlabel("Fraction of Variables Removed")
plt.ylabel("RGE")
plt.title("Voting Ensemble Model RGE Curve (New York Original)")
plt.legend()
plt.grid(True)
plt.savefig("plots/VEM_RGE_ny_original.png", dpi=300)
plt.close()

# Robustness
# RGR AUC
thresholds = np.arange(0, 0.51, 0.01)
results = [rgr_all(x_test, y_prob, voting_clf, t) for t in thresholds]
normalized_thresholds = thresholds / 0.5
rgr_auc = auc(normalized_thresholds, results)
print(f"AURGR: {rgr_auc:.4f}")

# Plot
plt.figure(figsize=(6, 4))
plt.plot(normalized_thresholds, results, linestyle='-', label=f"RGR Curve (AURGR = {rgr_auc:.4f})")
plt.title('Voting Ensemble Model RGR Curve (New York Original)')
plt.axhline(0.5, color='red', linestyle='--', label="Random Classifier (RGR = 0.5)")
plt.xlabel('Normalized Perturbation')
plt.ylabel('RGR')
plt.legend()
plt.xlim([0, 1])
plt.grid(True)
plt.savefig("plots/VEM_RGR_ny_original.png", dpi=300)
plt.close()

# Fairness
fair = compute_rge_values(x_train, x_test, y_prob, voting_clf, ["applicant_sex", "applicant_race_1"])
print(fair)

# Values for final metric
# RGA part
num_steps = len(step_rges) - 1
step_rgas = []
thresholds_rga = np.linspace(1, 0, num_steps + 1)
for i in range(num_steps):
    lower = float(thresholds_rga[i + 1])
    upper = float(thresholds_rga[i])
    partial = partial_rga_with_curves(y_test, y_prob, lower, upper, False)
    step_rgas.append(partial)
reverse_cumulative = np.cumsum(step_rgas[::-1])[::-1]
x_final = np.concatenate((reverse_cumulative, [0.])).tolist()

# RGE part
y_final = step_rges

# RGR part
num_steps_rgr = len(step_rges)
thresholds_rgr = np.linspace(0, 0.5, num_steps_rgr)
z_final = [rgr_all(x_test, y_prob, voting_clf, t) for t in thresholds_rgr]

# Save results
data = {
    "x_final": x_final,
    "y_final": y_final,
    "z_final": z_final
}
json_str = json.dumps(data, indent=4)
file_path = os.path.join("../saved_data", "final_results_voting_ny_original.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)