import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import TomekLinks
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from check_explainability import compute_rge_values
from check_fairness import compute_rga_parity
from check_robustness import compute_rgr_values
from core import rga
from data_processing_credits import data_lending_clean

# Data separation
x = data_lending_clean.iloc[:, :-1]  # Features
y = data_lending_clean.iloc[:, -1]  # Target (0 = approved, 1 = rejected)

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Standardize features
scaler = StandardScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

# PCA Decomposition on both train and test sets
pca = PCA(n_components=2)
pca.fit(x_train_scaled)
x_train_pca = pca.transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

# Plot dataset
plt.figure(figsize=(5, 5))
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, alpha=0.5, s=30, edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Original Dataset')
plt.show()

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(x_train_scaled, y_train)

# Make predictions
y_pred = log_model.predict(x_test_scaled)
y_prob = log_model.predict_proba(x_test_scaled)[:, 1]  # Probabilities for the positive class

# AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC Score:", roc_auc)

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
print(f'Best F1 Score: {best_f1["f1_score"]:.4f} at Threshold: {best_threshold:.4f}')

# Predictions at best threshold
y_pred_best = np.where(y_prob >= best_threshold, 1, 0)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Tomek Links undersampling and PCA
tl = TomekLinks()
x_train_resampled, y_train_resampled = tl.fit_resample(x_train_scaled, y_train)
x_train_resampled_pca = pca.transform(x_train_resampled)

# Plot original dataset and after resampling
plt.figure(figsize=(6, 6))
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, alpha=0.5, s=30, edgecolor='k', label='Original Dataset')
plt.scatter(x_train_resampled_pca[:, 0], x_train_resampled_pca[:, 1], c=y_train_resampled, alpha=0.5, s=30, edgecolor='r', marker='x', label='After Tomek Links')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Original vs. Resampled Dataset')
plt.legend()
plt.show()

# Again Logistic Regression
log_tomek_model = LogisticRegression()
log_tomek_model.fit(x_train_resampled, y_train_resampled)

# New predictions
y_pred_tomek = log_tomek_model.predict(x_test_scaled)
y_prob_tomek = log_tomek_model.predict_proba(x_test_scaled)[:, 1]  # Probabilities for the positive class

# ROC Curve
fpr_tomek, tpr_tomek, thresholds_tomek = roc_curve(y_test, y_prob_tomek)
roc_auc_tomek = auc(fpr_tomek, tpr_tomek)
print("AUC Score:", roc_auc_tomek)

# Partial AUC
fpr_threshold_tomek = 0.3
valid_indices_tomek = np.where(fpr_tomek < fpr_threshold_tomek)[0]
partial_auc_tomek = auc(fpr_tomek[valid_indices_tomek], tpr_tomek[valid_indices_tomek]) / fpr_threshold_tomek
print(f"Partial AUC (FPR < {fpr_threshold_tomek}): {partial_auc_tomek:.4f}")

# Plotting ROC curve
plt.figure()
plt.plot(fpr_tomek, tpr_tomek, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_tomek:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()

# DataFrame with thresholds, FPR, TPR, and F1 Score
roc_data_tomek = pd.DataFrame({
    'threshold': thresholds_tomek,
    'fpr': fpr_tomek,
    'tpr': tpr_tomek,
    'f1_score': [f1_score(y_test, np.where(y_prob_tomek >= t, 1, 0)) for t in thresholds_tomek]
})

# Best F1 score and corresponding threshold
best_f1_tomek = roc_data_tomek.loc[roc_data_tomek['f1_score'].idxmax()]
best_threshold_tomek = best_f1_tomek["threshold"]
print(f'Best F1 Score: {best_f1_tomek["f1_score"]:.4f} at Threshold: {best_threshold_tomek:.4f}')

# Predictions at best threshold
y_pred_best_tomek = np.where(y_prob_tomek >= best_threshold_tomek, 1, 0)

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred_best_tomek))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best_tomek))
print("Classification Report:\n", classification_report(y_test, y_pred_best_tomek))

# Integrating safeai
# Accuracy
rga_class = rga(y_test, y_prob_tomek)
print(f"RGA value is equal to {rga_class}")

# Explainability
print(compute_rge_values(x_train_resampled, x_test_scaled, y_prob_tomek, log_tomek_model, ["loan_purpose"]))

# Fairness
print(compute_rga_parity(x_train_resampled, x_test_scaled, y_test, y_prob_tomek, log_tomek_model, "applicant_race_1"))

# Robustness
print(compute_rgr_values(x_test_scaled, y_prob_tomek, log_tomek_model, list(x_test_scaled.columns)))