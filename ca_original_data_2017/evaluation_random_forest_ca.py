import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_curve

from safeai_files.check_explainability import compute_rge_values
from safeai_files.check_fairness import compute_rga_parity
from safeai_files.check_robustness import compute_rgr_values, rgr_all, rgr_single
from safeai_files.core import rga

# Load best model and variables
best_model = joblib.load("../saved_models/best_rf_model_ca.joblib")
x_train = pd.read_csv("../saved_data/x_train_rf_ca.csv")
x_test = pd.read_csv("../saved_data/x_test_rf_ca.csv")
y_train = pd.read_csv("../saved_data/y_train_rf_ca.csv")
y_test = pd.read_csv("../saved_data/y_test_rf_ca.csv")
y_test = y_test.values.tolist()

# Make predictions
y_pred = best_model.predict(x_test)
y_prob = best_model.predict_proba(x_test)[:, 1]

# AUC (for future use)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(roc_auc)

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

# Integrating safeai
# Accuracy
rga_class = rga(y_test, y_prob)
print(f"RGA value is equal to {rga_class}")

# Explainability
print(compute_rge_values(x_train, x_test, y_prob, best_model, ["loan_purpose"]))

# Fairness
print(compute_rga_parity(x_train, x_test, y_test, y_prob, best_model, "applicant_sex"))
print(compute_rga_parity(x_train, x_test, y_test, y_prob, best_model, "applicant_race_1"))

# Robustness
print(compute_rgr_values(x_test, y_prob, best_model, list(x_test.columns)))
print(rgr_all(x_test, y_prob, best_model, 0.2))
print(rgr_single(x_test, y_prob, best_model, "loan_purpose", 0.2))