import json
import itertools
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ca_original_data_2017.data_processing_credits_ca import data_lending_ca_clean
from torch_for_credits.torch_model import CreditModel

# Data separation
x = data_lending_ca_clean.drop(columns=['action_taken'])
y = data_lending_ca_clean['action_taken']

# Splitting into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

#GPU connection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Ensure cuDNN is optimized
torch.backends.cudnn.benchmark = True

# Standardize data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test) #important not to fit again(mean and standard deviation)
x_train_scaled_names = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
x_test_scaled_names = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) #1D (n) -> 2D (n,1)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Hyperparameters tuning
batch_sizes = [64, 128, 256]
max_lr = 0.01
hidden_layer_sizes = [(32,16), (64, 32, 16), (128, 64, 32)]
dropout_rates = [0.2, 0.3]

# Preparation
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
best_accuracy = 0
best_params = None
best_model_path = None
loss_values = []
model = None
model_dir = "../saved_models"
os.makedirs(model_dir, exist_ok=True) # Create directory if it doesn't exist

# Automatic mixed precision
scaler_amp = torch.amp.GradScaler(device="cuda")

# Learning
for batch_size, layers, dropout_rate in itertools.product(batch_sizes, hidden_layer_sizes,
                                                              dropout_rates):
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
        x_train_fold, x_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Standardize each fold
        x_train_fold_scaled = scaler.transform(x_train_fold)
        x_val_fold_scaled = scaler.transform(x_val_fold)

        # Convert to tensors
        x_train_fold_tensor = torch.tensor(x_train_fold_scaled, dtype=torch.float32, device=device)
        y_train_fold_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32, device=device).view(-1, 1)
        x_val_fold_tensor = torch.tensor(x_val_fold_scaled, dtype=torch.float32, device=device)
        y_val_fold_tensor = torch.tensor(y_val_fold.values, dtype=torch.float32, device=device).view(-1, 1)

        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(x_train_fold_tensor, y_train_fold_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Define model
        model = CreditModel(x_train.shape[1], layers, dropout_rate).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters())

        # 1Cycle Learning Rate
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=50)

        # Early Stopping Setup
        patience = 5
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(50):  # Train for up to 50 epochs
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)  # Move batch to GPU
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):  # Use AMP for mixed precision
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
                if counter < patience:
                    scheduler.step() #Update Learning rate

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            loss_values.append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_fold_tensor)
                val_loss = criterion(val_outputs, y_val_fold_tensor).item()

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Evaluate on validation set
        with torch.no_grad():
            y_val_pred = model(x_val_fold_tensor)
            y_val_pred_labels = (y_val_pred > 0.5).float()
            val_accuracy = accuracy_score(y_val_fold_tensor.cpu().numpy(), y_val_pred_labels.cpu().numpy())
            fold_accuracies.append(val_accuracy)

    avg_fold_accuracy = np.mean(fold_accuracies)

    # Track best model
    if avg_fold_accuracy > best_accuracy:
        best_accuracy = avg_fold_accuracy
        best_params = (batch_size, max_lr, layers, dropout_rate)
        best_model_probabilities = y_val_pred.cpu().numpy()

        # Save the best model
        best_torch_model_path = os.path.join(model_dir, "best_credit_model_ca.pth")
        if model is not None:
            torch.save(model.state_dict(), best_torch_model_path)
            print(f"New best model saved at {best_model_path}")
        else:
            print("Model was not initialized")

print("\nBest Hyperparameters:", best_params, "with Accuracy:", best_accuracy)
print(f"Best model saved at {best_model_path}")

# Save variables to the folder
# Tensors
tensor_path = os.path.join("../saved_data", "full_data_tensors_ca.pth")
torch.save({
    "x_train_tensor": x_train_tensor,
    "y_train_tensor": y_train_tensor,
    "x_test_tensor": x_test_tensor,
    "y_test_tensor": y_test_tensor,
}, tensor_path)

# Best parameters
json_str = json.dumps(best_params, indent=4)
file_path = os.path.join("../saved_data", "best_torch_params_ca.json")
with open(file_path, "w", encoding="utf-8") as file:
    file.write(json_str)
print("Best parameters saved successfully!")

# Other
x_train_scaled_names.to_csv(os.path.join("../saved_data", "x_train_scaled_names_ca.csv"), index=False)
x_test_scaled_names.to_csv(os.path.join("../saved_data", "x_test_scaled_names_ca.csv"), index=False)
np.save(os.path.join("../saved_data", "y_train_ca.npy"), y_train)
np.save(os.path.join("../saved_data", "y_test_ca.npy"), y_test)
np.save(os.path.join("../saved_data", "loss_values_ca.npy"), np.array(loss_values))