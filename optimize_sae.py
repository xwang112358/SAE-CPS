import optuna
import torch
from torch.utils.data import DataLoader
from SAE import SparseAutoEncoder
import numpy as np
import csv
import os
from datetime import datetime
from tqdm import tqdm

def objective(trial, train_loader, val_loader, input_dim, device):
    # Define hyperparameters to optimize
    dict_size_multiplier = trial.suggest_categorical("dict_size_multiplier", [2, 4, 8])
    dict_size = dict_size_multiplier * input_dim
    
    # Sparse lambda range: focusing more on moderate sparsity (1e-3 to 1e-2)
    # while still allowing exploration of lighter (1e-4) and stronger (1e-1) sparsity
    sparse_lambda = trial.suggest_float("sparse_lambda", 1e-3, 1e-2, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    # Create model
    model = SparseAutoEncoder(input_dim=input_dim, dict_size=dict_size, lambda_sparse=sparse_lambda)
    model = model.to(device)  
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Train model
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc="Training", total=num_epochs):
        # Training
        model.train()
        train_losses = []
        for input_X in train_loader:
            input_X = input_X.to(device)  # Move input to device
            _, _, loss, _, _ = model.forward_pass(input_X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for input_X in val_loader:
                input_X = input_X.to(device)  # Move input to device
                _, _, _, val_recon_loss, _ = model.forward_pass(input_X)
                val_losses.append(val_recon_loss.item())
        
        mean_val_loss = np.mean(val_losses)
        
        # Update best validation loss
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
        
    with open(f"./training_logs/sae/fine_tuning_results.csv", "a") as f:
        writer = csv.writer(f)
        # Add header if file is empty
        if f.tell() == 0:
            writer.writerow(['trial_number', 'validation_loss', 'dict_size', 'learning_rate', 'sparse_lambda'])
        writer.writerow([trial.number, best_val_loss, input_dim * dict_size_multiplier, learning_rate, sparse_lambda])
    
    return best_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = torch.load("./dataset/BATADAL_train_dataset.pt")
    val_dataset = torch.load("./dataset/BATADAL_val_dataset.pt")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_dim = train_dataset.shape[1]
    
    study = optuna.create_study(
        direction="minimize",
        pruner=None
    )
    
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, input_dim, device), 
        n_trials=50, 
        n_jobs=10,  
        show_progress_bar=True
    )
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    os.makedirs("./training_logs/sae", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = f"./training_logs/sae/best_hyperparameters.csv"

    # Write hyperparameters to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'dict_size_multiplier', 'learning_rate', 'sparse_lambda', 'best_validation_loss'])
        writer.writerow([timestamp, trial.params['dict_size_multiplier'], trial.params['learning_rate'], trial.params['sparse_lambda'], trial.value])

    print(f"\nBest hyperparameters saved to: {csv_path}")

    # run the best hyperparameters
    dict_size_multiplier = trial.params["dict_size_multiplier"]
    dict_size = dict_size_multiplier * input_dim
    sparse_lambda = trial.params["sparse_lambda"]
    learning_rate = trial.params["learning_rate"]
    
    model = SparseAutoEncoder(input_dim=input_dim, dict_size=dict_size, lambda_sparse=sparse_lambda).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_epochs = 200
    
    model.train_model(train_loader, val_loader, optimizer, num_epochs)
    


if __name__ == "__main__":
    main() 