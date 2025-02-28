import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve
from safeaipackage.check_explainability import compute_rge_values
from xgboost import XGBClassifier

from main import data_lending_clean

# Load and preprocess data
x = data_lending_clean.iloc[:, :-1]  # Features
y = data_lending_clean.iloc[:, -1]  # Target (0 = approved, 1 = rejected)

# Splitting into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Standardize features
scaler = StandardScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

# Logistic Regression Model
model_log = LogisticRegression()
model_log.fit(x_train_scaled, y_train)

# Predictions
y_pred = model_log.predict(x_test_scaled)
y_prob = model_log.predict_proba(x_test_scaled)[:, 1]  # Probabilities for the positive class

# Evaluate Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(roc_auc)

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

variable_names = x_train.columns.tolist()
explain = compute_rge_values(x_train, x_test, y_pred, model_log, variables = variable_names)
print(explain)