import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from core import rga
from check_explainability import compute_rge_values
from check_fairness import compute_rga_parity
from check_robustness import compute_rgr_values
import json

from main import data_lending_clean

x = data_lending_clean.iloc[:, :-1]  # Features
y = data_lending_clean.iloc[:, -1]  # Target (0 = approved, 1 = rejected)

# Splitting into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Grid search
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 15],
    'max_features': [5, int(x_train.shape[1] / 2), 'sqrt', 'log2', x_train.shape[1]],
    'min_samples_leaf': [1, 2, 4, 8]
}
grid_search = GridSearchCV(RF(random_state=5), param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=3)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print(f"Best max_features: {best_params}")
print(f"Best F1 Score from CV: {grid_search.best_score_:.4f}")

# Save best parameters
json_str = json.dumps(best_params, indent=4)
with open("best_rf_params.json", "w", encoding="utf-8") as file:
    file.write(json_str)
print("Best parameters saved successfully!")

#Random Forest
random_forest = RF(**best_params, random_state=15)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
y_prob = random_forest.predict_proba(x_test)[:, 1]  # Probabilities for the positive class
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#AUC
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
print(f'Best F1 Score: {best_f1["f1_score"]:.4f} at Threshold: {best_threshold:.4f}')

# Predictions at best threshold
y_pred_best = np.where(y_prob >= best_threshold, 1, 0)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Integrating safeai
# Accuracy
rga_class = rga(y_test, y_prob)
print(f"RGA value is equal to {rga_class}")

# Explainability
print(compute_rge_values(x_train, x_test, y_prob, random_forest, ["loan_purpose"]))

# Fairness
print(compute_rga_parity(x_train, x_test, y_test, y_prob, random_forest, "applicant_race_1"))

# Robustness
print(compute_rgr_values(x_test, y_prob, random_forest, list(x_test.columns)))