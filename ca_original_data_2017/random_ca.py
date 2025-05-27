import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

from safeai_files.check_explainability import compute_rge_values
from safeai_files.check_robustness import rgr_all
from safeai_files.core import partial_rga_with_curves, rga

# Data separation
data_lending_ca_clean = pd.read_csv("../saved_data/data_lending_clean_ca.csv")
predictors = data_lending_ca_clean.drop(columns=['action_taken'])
y = data_lending_ca_clean['action_taken']

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(predictors, y, test_size=0.2, random_state=15)
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)

# Random Classifier
random_model = DummyClassifier(strategy='prior', random_state=42)

# Train
random_model.fit(x_train, y_train)

# Make predictions
y_pred = random_model.predict(x_test)
y_prob = random_model.predict_proba(x_test)[:, 1]

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
        rge_k = compute_rge_values(x_train, x_test, y_prob, random_model, current_vars, group=True)
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
plt.xlabel("Fraction of Variables Removed")
plt.ylabel("RGE")
plt.title("Random Classifier RGE Curve (California)")
plt.legend()
plt.grid(True)
plt.savefig("plots/Random_RGE_ca.png", dpi=300)
plt.close()

# Robustness
# RGR AUC
thresholds = np.arange(0, 0.51, 0.01)
results = [rgr_all(x_test, y_prob, random_model, t) for t in thresholds]
normalized_thresholds = thresholds / 0.5
rgr_auc = auc(normalized_thresholds, results)
print(f"AURGR: {rgr_auc:.4f}")

# Plot
plt.figure(figsize=(6, 4))
plt.plot(normalized_thresholds, results, linestyle='-', label=f"RGR Curve (AURGR = {rgr_auc:.4f})")
plt.title('Random Classifier RGR Curve (California)')
plt.xlabel('Normalized Perturbation')
plt.ylabel('RGR')
plt.legend()
plt.xlim([0, 1])
plt.grid(True)
plt.savefig("plots/Random_RGR_ca.png", dpi=300)
plt.close()

# Fairness
fair = compute_rge_values(x_train, x_test, y_prob, random_model, ["applicant_sex", "applicant_race_1"])
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
z_final = [rgr_all(x_test, y_prob, random_model, t) for t in thresholds_rgr]

# Save results
data = {
    "x_final": x_final,
    "y_final": y_final,
    "z_final": z_final
}
json_str = json.dumps(data, indent=4)
file_path = os.path.join("../saved_data", "final_results_random_ca.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)