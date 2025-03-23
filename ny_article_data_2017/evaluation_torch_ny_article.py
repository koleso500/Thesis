import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

from safeai_files.check_explainability import compute_rge_values
from safeai_files.check_fairness import compute_rga_parity
from safeai_files.check_robustness import compute_rgr_values
from safeai_files.core import rga
from torch_for_credits.torch_model import CreditModel

# Directory of the data
save_dir = "../saved_data"

# Load torch tensors
tensor_path = os.path.join(save_dir, "full_data_tensors_ny_article.pth")
tensor_data = torch.load(tensor_path)
x_train_tensor = tensor_data["x_train_tensor_ny_article"]
y_train_tensor = tensor_data["y_train_tensor_ny_article"]
x_test_tensor = tensor_data["x_test_tensor_ny_article"]
y_test_tensor = tensor_data["y_test_tensor_ny_article"]

# Load best torch parameters
best_params_path = os.path.join(save_dir, "best_torch_params_ny_article.json")
with open(best_params_path, "r", encoding="utf-8") as file:
    best_params = json.load(file)

# Load other data
x_train_scaled_names = pd.read_csv(os.path.join(save_dir, "x_train_scaled_names_ny_article.csv"))
x_test_scaled_names = pd.read_csv(os.path.join(save_dir, "x_test_scaled_names_ny_article.csv"))
y_train = np.load(os.path.join(save_dir, "y_train_ny_article.npy"))
y_test = np.load(os.path.join(save_dir, "y_test_ny_article.npy"))

# Load loss values
loss_values = np.load(os.path.join(save_dir, "loss_values_ny_article.npy"))

def evaluate_model_ny_article():
    # Load the best model
    best_model = CreditModel(x_train_scaled_names.shape[1], best_params[2], best_params[3])
    best_model.load_state_dict(torch.load('../saved_models/best_credit_model_ny_article.pth', weights_only=True))
    best_model.eval()

    # Make predictions on the test set
    with torch.no_grad():
        y_pred = best_model(x_test_tensor)
        y_pred_labels = (y_pred > 0.5).float()

    # Convert to numpy for evaluation
    y_test_numpy = y_test_tensor.cpu().numpy()
    y_pred_prob = y_pred.cpu().numpy()

    # Evaluate model performance
    print("Accuracy:", accuracy_score(y_test_tensor.numpy(), y_pred_labels.numpy()))
    print("Confusion Matrix:\n", confusion_matrix(y_test_numpy, y_pred_labels))
    print("Classification Report:\n", classification_report(y_test_numpy, y_pred_labels))

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid()
    plt.show()

    # Integrating safeai
    # Accuracy
    rga_class = rga(y_test, y_pred_prob)
    print(f"RGA value is equal to {rga_class}")

    # Explainability
    print(compute_rge_values(x_train_scaled_names, x_test_scaled_names, y_pred_prob, best_model, ["loan_purpose"]))

    # Fairness
    print(compute_rga_parity(x_train_scaled_names, x_test_scaled_names, y_test, y_pred_prob, best_model, "applicant_race_1"))

    # Robustness
    print(compute_rgr_values(x_test_scaled_names, y_pred_prob, best_model, list(x_test_scaled_names.columns)))

if __name__ == "__main__":
    evaluate_model_ny_article()