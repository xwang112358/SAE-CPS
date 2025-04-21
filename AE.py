import torch
import torch.nn as nn
import os
import numpy as np
import datetime
import pickle
# D = d_model (input dimension)
# F = hidden_dim (latent dimension)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    """
    A one-layer vanilla autoencoder.
    """
    def __init__(self, 
                 input_dim: int, 
                 dict_size: int):
        
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size

        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)

    def encode(self, 
               input_X: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        Args:
            input_X: Input tensor of shape (batch_size, input_dim)
        Returns:
            Encoded representation of shape (batch_size, dict_size)
        """
        return nn.ReLU()(self.encoder(input_X))
    
    def decode(self, 
               encoded_representation: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        Args:
            encoded_representation: Latent tensor of shape (batch_size, dict_size)
        Returns:
            Reconstructed input of shape (batch_size, input_dim)
        """
        return self.decoder(encoded_representation)
    
    def forward_pass(self, 
                     input_X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Args:
            input_X: Input tensor of shape (batch_size, input_dim)
        Returns:
            Tuple of (reconstructed_input, encoded_representation, reconstruction_loss)
        """
        encoded_representation = self.encode(input_X)
        reconstructed_input_X = self.decode(encoded_representation)
        
        # Compute reconstruction loss using MSE
        recon_loss = torch.mean(torch.sum((reconstructed_input_X - input_X) ** 2, dim=1))
        
        return reconstructed_input_X, encoded_representation, recon_loss

    def train_model(self, 
              train_loader: torch.utils.data.DataLoader,
              valid_loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              num_epochs: int):
        """
        Train the autoencoder.
        Args:
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            optimizer: Optimizer for training
            num_epochs: Number of epochs to train
        Returns:
            Dictionary containing training and validation loss history
        """
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        best_valid_loss = float("inf")
        best_epoch = 0

        loss_history = {"train": [], "valid": []}

        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            valid_losses = []
            
            for input_X in train_loader:
                input_X = input_X.to(device)
                reconstructed_input_X, encoded_representation, train_loss = self.forward_pass(input_X)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.item())
            
            mean_train_loss = np.mean(train_losses)
            print(f"Epoch {epoch}, Train Loss: {mean_train_loss}")
            loss_history["train"].append(mean_train_loss)

            self.eval()
            with torch.no_grad():
                for input_X in valid_loader:
                    input_X = input_X.to(device)
                    reconstructed_input_X, encoded_representation, valid_loss = self.forward_pass(input_X)
                    valid_losses.append(valid_loss.item())

            mean_valid_loss = np.mean(valid_losses)
            print(f"Epoch {epoch}, Valid Loss: {mean_valid_loss}")
            loss_history["valid"].append(mean_valid_loss)

            if mean_valid_loss < best_valid_loss:
                best_valid_loss = mean_valid_loss
                best_epoch = epoch
                save_path = f"./checkpoints/ae/{date_time}"
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.state_dict(), f"{save_path}/best_epoch.pth")
                print(f"Saved model with lowest valid loss: {best_valid_loss} at epoch {best_epoch}")

        # save loss history in pkl
        with open(f"{save_path}/loss_history.pkl", "wb") as f:
            pickle.dump(loss_history, f)
        


        return loss_history

    def load_model(self,
                   model_path: str):
        """Load a pretrained model."""
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
    
    def predict_anomaly(self,
                        model_path: str,
                        input_X: torch.Tensor,
                        quantile: float = 0.95,
                        normalize: bool = True,
                        window: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomalies using reconstruction error and quantile threshold with window-based smoothing.
        Args:
            model_path: Path to the saved model file
            input_X: Input tensor to predict anomalies on
            quantile: Quantile threshold for anomaly detection (default: 0.95)
            normalize: Whether to normalize reconstruction errors (default: True)
            window: Size of the sliding window for smoothing (default: 1)
        Returns:
            Tuple containing:
            - Reconstruction error for each instance
            - Binary labels indicating anomalies (1) and normal samples (0)
        """
        self.load_model(model_path)
        # print(f"Loaded model from {model_path}")
        
        self.eval()
        with torch.no_grad():
            reconstructed_input_X, encoded_representation, _ = self.forward_pass(input_X)
            # Calculate reconstruction errors
            reconstruction_error = torch.sum((reconstructed_input_X - input_X) ** 2, dim=1)
            
            if normalize:
                # Normalize errors to have zero mean and unit variance
                mean_error = torch.mean(reconstruction_error)
                std_error = torch.std(reconstruction_error)
                reconstruction_error = (reconstruction_error - mean_error) / std_error

            # Apply window-based smoothing if window > 1
            if window > 1:
                # Convert to numpy for rolling window operation
                errors_np = reconstruction_error.cpu().numpy()
                
                # Apply moving average
                errors_smoothed = np.convolve(errors_np, np.ones(window)/window, mode='valid')
                
                # Pad the beginning to maintain the same length
                padding = np.full(window-1, errors_smoothed[0])
                errors_smoothed = np.concatenate([padding, errors_smoothed])
                reconstruction_error = torch.tensor(errors_smoothed, device=reconstruction_error.device)
            
            # Calculate threshold using quantile
            threshold = torch.quantile(reconstruction_error, quantile)
            
            # Generate anomaly labels
            anomaly_labels = (reconstruction_error > threshold).float()
            
            return reconstruction_error, anomaly_labels

    def calculate_feature_contribution(self, 
                                       input_X: torch.Tensor,
                                       model_path: str,
                                       decoder_only: bool = False) -> torch.Tensor:
        """
        Calculate contribution score Cⱼ for each input feature j.
        
        Args:
            input_X: Input tensor to analyze
            model_path: Path to the saved model file
            decoder_only: If True, use only decoder weights for contribution calculation
        Returns:
            Tensor containing contribution score for each feature
        """
        self.load_model(model_path)
        self.eval()
        with torch.no_grad():
            # Get encoder and decoder weights
            W_e = torch.abs(self.encoder.weight)  # Shape: (dict_size, input_dim)
            W_d = torch.abs(self.decoder.weight)  # Shape: (input_dim, dict_size)
            
            z = self.encode(input_X)  # Shape: (batch_size, dict_size)
            
            contributions = torch.zeros(self.input_dim, device=input_X.device) # Ensure tensor is on the same device

            if decoder_only:
                for j in range(self.input_dim):
                    for k in range(self.dict_size):
                        contributions[j] += W_d[j,k] * z[0,k] # Assuming batch size of 1 for analysis
            else:
                for j in range(self.input_dim):
                    for k in range(self.dict_size):
                        contributions[j] += W_e[k,j] * z[0,k] * W_d[j,k] # Assuming batch size of 1 for analysis
                    
            return contributions


if __name__ == "__main__":
    # Example usage
    input_dim = 768
    dict_size = 256
    
    ae = AutoEncoder(input_dim, dict_size)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=0.001)
    
    # Assuming you have your data loaders set up
    # train_loader = ...
    # valid_loader = ...
    # loss_history = ae.train_model(train_loader, valid_loader, optimizer, num_epochs=100)
    
    # For analyzing a single sample
    # sample = torch.randn(1, input_dim)
    # analysis = ae.analyze_anomaly(
    #     sample,
    #     model_path="./checkpoints/your_model.pth",
    #     top_n_features=5
    # )