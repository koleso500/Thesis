import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from ca_original_data_2017.data_processing_ca import data_lending_ca_clean

# Data separation
x = data_lending_ca_clean.drop(columns=['action_taken'])
y = data_lending_ca_clean['action_taken']

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Save the splits
x_train.to_csv(os.path.join("../saved_data", "x_train_rf_ca.csv"), index=False)
x_test.to_csv(os.path.join("../saved_data", "x_test_rf_ca.csv"), index=False)
y_train.to_csv(os.path.join("../saved_data", "y_train_rf_ca.csv"), index=False)
y_test.to_csv(os.path.join("../saved_data", "y_test_rf_ca.csv"), index=False)

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameters grid
params_grid = {
    'n_estimators': [500], #100,200,300
    'max_depth': [10], #None, 5, 15
    'max_features': ['sqrt'], #'log2', x_train.shape[1], 5, int(x_train.shape[1] / 2),
    'min_samples_leaf': [8] #1,2,4
}

# Grid search
grid_search = GridSearchCV(rf_model, params_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=3)
grid_search.fit(x_train, y_train)

# Best model, best parameters and AUC
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
print("Best AUC:", grid_search.best_score_)

# Save best parameters and model
json_str = json.dumps(best_params, indent=4)
file_path = os.path.join("../saved_data", "best_rf_params_ca.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)
print("Best parameters saved successfully!")

model_path = os.path.join("../saved_models", "best_rf_model_ca.joblib")
joblib.dump(best_model, model_path)