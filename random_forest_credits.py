import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from check_explainability import compute_rge_values
from check_fairness import compute_rga_parity
from check_robustness import compute_rgr_values
from core import rga
from data_processing_credits import data_lending_clean

# Data separation
x = data_lending_clean.iloc[:, :-1]  # Features
y = data_lending_clean.iloc[:, -1]  # Target (0 = approved, 1 = rejected)

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameters grid
params_grid = {
    'n_estimators': [100, 200, 300, 500], 
    'max_depth': [None, 5, 10, 15],
    'max_features': [5, int(x_train.shape[1] / 2), 'sqrt', 'log2', x_train.shape[1]],
    'min_samples_leaf': [1, 2, 4, 8]
}

# Grid search
grid_search = GridSearchCV(rf_model, params_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=3)
grid_search.fit(x_train, y_train)

# Best model, best parameters and AUC
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
print("Best AUC:", grid_search.best_score_)

# Save best parameters
json_str = json.dumps(best_params, indent=4)
file_path = os.path.join("saved_data", "best_rf_params.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)
print("Best parameters saved successfully!")

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

# SHAP plot
explainer = shap.Explainer(best_model)
shap_values = explainer(x_test)
shap.summary_plot(shap_values, x_test)

# Integrating safeai
# Accuracy
rga_class = rga(y_test, y_prob)
print(f"RGA value is equal to {rga_class}")

# Explainability
print(compute_rge_values(x_train, x_test, y_prob, best_model, ["loan_purpose"]))

# Fairness
print(compute_rga_parity(x_train, x_test, y_test, y_prob, best_model, "applicant_race_1"))

# Robustness
print(compute_rgr_values(x_test, y_prob, best_model, list(x_test.columns)))