import joblib
import json
import os
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb

from ny_original_data_2017.data_processing_ny_original import data_lending_ny_clean

# Data separation
x = data_lending_ny_clean.drop(columns=['action_taken'])
y = data_lending_ny_clean['action_taken']

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Save the splits
x_train.to_csv(os.path.join("../saved_data", "x_train_xgb_ny_original.csv"), index=False)
x_test.to_csv(os.path.join("../saved_data", "x_test_xgb_ny_original.csv"), index=False)
y_train.to_csv(os.path.join("../saved_data", "y_train_xgb_ny_original.csv"), index=False)
y_test.to_csv(os.path.join("../saved_data", "y_test_xgb_ny_original.csv"), index=False)

# XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

# Hyperparameters grid
params_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1],
    'gamma': [0, 0.1, 0.2]
}

# Grid search
grid_search = GridSearchCV(xgb_model, params_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=3)
grid_search.fit(x_train, y_train)

# Best model, best parameters and AUC
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
print("Best AUC:", grid_search.best_score_)

# Save best parameters and model
json_str = json.dumps(best_params, indent=4)
file_path = os.path.join("../saved_data", "best_xgb_params_ny_original.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)
print("Best parameters saved successfully!")

model_path = os.path.join("../saved_models", "best_xgb_model_ny_original.joblib")
joblib.dump(best_model, model_path)