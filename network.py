import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import pandas as pd
import os
import itertools
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import json

from main import data_lending_clean
from model import CreditModel

# Load and preprocess data
x = data_lending_clean.iloc[:, :-1]  # Features
y = data_lending_clean.iloc[:, -1]  # Target (0 = approved, 1 = rejected)

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
initial_scaler = StandardScaler()
x_train_scaled = initial_scaler.fit_transform(x_train)
x_test_scaled = initial_scaler.transform(x_test) #important not to fit again(mean and standard deviation)
x_train_scaled_names = pd.DataFrame(initial_scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
x_test_scaled_names = pd.DataFrame(initial_scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) #1D (n) -> 2D (n,1)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Hyperparameter tuning
batch_sizes = [128] #64,128,256
max_lr = 0.01
hidden_layer_sizes = [(64, 32, 16), (128, 64, 32)] #(32,16)
dropout_rates = [0.2] #0.3

k_folds = 3 #5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

best_accuracy = 0
best_params = None
best_model_path = None
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True) # Create directory if it doesn't exist

scaler = torch.amp.GradScaler(device="cuda")

for batch_size, layers, dropout_rate in itertools.product(batch_sizes, hidden_layer_sizes,
                                                              dropout_rates):
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
        x_train_fold, x_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Standardize each fold
        data_scaler = StandardScaler()
        x_train_fold_scaled = data_scaler.fit_transform(x_train_fold)
        x_val_fold_scaled = data_scaler.transform(x_val_fold)

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
        loss_values = []

        for epoch in range(50):  # Train for up to 50 epochs
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)  # Move batch to GPU
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):  # Use AMP for mixed precision
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
        best_model_path = os.path.join(model_dir, "best_credit_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at {best_model_path}")

print("\nBest Hyperparameters:", best_params, "with Accuracy:", best_accuracy)
print(f"Best model saved at {best_model_path}")

# Save necessary variables
with open("saved_data.pkl", "wb") as f:
    pickle.dump({
        "x_train": x_train,
        "x_train_scaled_names": x_train_scaled_names,
        "x_test_scaled_names": x_test_scaled_names,
        "x_test_tensor": x_test_tensor,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_tensor": y_test_tensor,
        "best_params": best_params,
        "loss_values": loss_values,
    }, f)

# Save necessary variables
with open("saved_data_torch.json", "w") as f:
    json.dump({
        "x_train": x_train,
        "x_train_scaled_names": x_train_scaled_names,
        "x_test_scaled_names": x_test_scaled_names,
        "x_test_tensor": x_test_tensor,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_tensor": y_test_tensor,
        "best_params": best_params,
        "loss_values": loss_values,
    }, f, default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj), indent=4)
print("Best parameters saved successfully!")