import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import pickle
from model import CreditModel
from core import rga
from check_explainability import compute_rge_values
from check_fairness import compute_rga_parity
from check_robustness import compute_rgr_values

with open("saved_data.pkl", "rb") as f:
    saved_data = pickle.load(f)
x_train = saved_data["x_train"]
x_train_scaled_names = saved_data["x_train_scaled_names"]
x_test_scaled_names = saved_data["x_test_scaled_names"]
x_test_tensor = saved_data["x_test_tensor"]
y_test_tensor = saved_data["y_test_tensor"]
y_train = saved_data["y_train"]
y_test = saved_data["y_test"]
best_params = saved_data["best_params"]
loss_values = saved_data["loss_values"]

def evaluate_model():
    # Load the Best Model
    best_model = CreditModel(x_train.shape[1], best_params[2], best_params[3])
    best_model.load_state_dict(torch.load('saved_models/best_credit_model.pth', weights_only=True))
    best_model.eval()

    # Make predictions on the test set
    with torch.no_grad():
        y_pred = best_model(x_test_tensor)
        y_pred_labels = (y_pred > 0.5).float()

    # Convert ground truth labels to NumPy for evaluation
    y_test_numpy = y_test_tensor.cpu().numpy()
    y_pred_prob = y_pred.cpu().numpy()

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test_numpy, y_pred_labels)

    # Compute F1-score
    f1 = f1_score(y_test_numpy, y_pred_labels)

    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nF1 Score:", f1)

    # Compute test accuracy
    test_accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_labels.numpy())
    print(f"Test Accuracy with Best Model: {test_accuracy:.4f}")

    # Plot Training Loss
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
    evaluate_model()