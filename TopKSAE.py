import torch
import torch.nn as nn
import os
import numpy as np
import datetime
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TopKSAE(nn.Module):
    """
    A k-sparse autoencoder that directly controls the number of active latents
    by using an activation function (TopK) that only keeps the k largest latents,
    zeroing the rest. Based on [Makhzani and Frey, 2013].
    """
    def __init__(self, 
                 input_dim: int, 
                 dict_size: int,
                 k: int):  # k represents how many latents to keep active
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)

    def topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the TopK activation function that keeps only the k largest values
        and zeros out the rest for each sample in the batch.
        
        Args:
            x: Input tensor of shape (batch_size, dict_size)
            
        Returns:
            Tensor with only k largest values per sample, rest zeroed out
        """
        if self.k >= x.shape[1]:  # if k is larger than feature dimension
            return x
            
        # Get the k largest values and their indices
        values, indices = torch.topk(x, k=self.k, dim=1)
        
        # Create a zero tensor of the same shape as input
        output = torch.zeros_like(x)
        
        # Fill in the k largest values at their original positions
        batch_indices = torch.arange(x.shape[0]).unsqueeze(1).expand(-1, self.k)
        output[batch_indices, indices] = values
        
        return output

    def encode(self, 
               input_X: torch.Tensor) -> torch.Tensor:
        """
        Implements z = TopK(W_enc(x - b_pre)) as per the paper
        """
        # Linear transformation
        z = self.encoder(input_X)
        
        # Apply TopK activation
        z = self.topk_activation(z)
        
        return z
    
    def decode(self, 
               encoded_representation: torch.Tensor) -> torch.Tensor:
        """
        Decodes the sparse representation back to input space
        """
        return self.decoder(encoded_representation)

    def forward_pass(self, 
                     input_X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        The training loss is simply L = ||x - x̂||²₂ as per the paper.
        """
        encoded_representation = self.encode(input_X)
        reconstructed_input_X = self.decode(encoded_representation)

        # Compute reconstruction loss as per paper: L = ||x - x̂||²₂
        recon_loss = torch.mean(torch.sum((reconstructed_input_X - input_X) ** 2, dim=1))
        
        # No explicit sparsity loss needed as sparsity is enforced by Top
        
        return reconstructed_input_X, encoded_representation, recon_loss
    
    def train_model(self, 
              train_loader: torch.utils.data.DataLoader,
              valid_loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              num_epochs: int):
        
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
                    reconstructed_input_X, encoded_representation, valid_recon_loss = self.forward_pass(input_X)
                    valid_losses.append(valid_recon_loss.item())

            mean_valid_loss = np.mean(valid_losses)
            print(f"Epoch {epoch}, Valid Loss: {mean_valid_loss}")
            loss_history["valid"].append(mean_valid_loss)

            # save the model with the lowest valid loss
            if mean_valid_loss < best_valid_loss:
                best_valid_loss = mean_valid_loss
                best_epoch = epoch

                save_path = f"./checkpoints/topk_sae/{date_time}"
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.state_dict(), f"{save_path}/best_epoch.pth")
                print(f"Saved model with lowest valid loss: {best_valid_loss} at epoch {best_epoch}")

        # save loss history in pkl
        with open(f"{save_path}/loss_history.pkl", "wb") as f:
            pickle.dump(loss_history, f)

        return loss_history
    
    def load_model(self,
                   model_path: str):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
    
    def predict_anomaly(self,
                        model_path: str,
                        input_X: torch.Tensor,
                        quantile: float = 0.95,
                        normalize: bool = True,
                        window: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load a pretrained model and use it for anomaly detection.
        Args:
            model_path: Path to the saved model file
            input_X: Input tensor to predict anomalies on (shape: N*M where N is batch size, M is feature dimension)
            quantile: Quantile threshold for anomaly detection (default: 0.95)
            normalize: Whether to normalize reconstruction errors (default: True)
            window: Size of the sliding window for smoothing (default: 1, no smoothing)
        Returns:
            Tuple containing:
            - Reconstruction error for each instance (shape: N)
            - Binary labels indicating anomalies (1) and normal samples (0)
        """
        self.load_model(model_path)
        # print(f"Loaded model from {model_path}")
        
        # Move to evaluation mode
        self.eval()
        
        with torch.no_grad():
            reconstructed_X, encoded_features, loss = self.forward_pass(input_X)
            reconstruction_error = torch.sum((reconstructed_X - input_X) ** 2, dim=1)
            
            # Normalize reconstruction errors if specified
            if normalize:
                mean_error = torch.mean(reconstruction_error)
                std_error = torch.std(reconstruction_error)
                # Add a small epsilon to prevent division by zero
                reconstruction_error = (reconstruction_error - mean_error) / (std_error + 1e-6)

            # Apply window-based smoothing if window > 1
            if window > 1:
                # Convert to numpy for rolling window operation
                errors_np = reconstruction_error.cpu().numpy()
                
                # Apply moving average
                errors_smoothed = np.convolve(errors_np, np.ones(window)/window, mode='valid')
                
                # Pad the beginning to maintain the same length
                # Handle case where errors_smoothed might be empty if window is too large
                if errors_smoothed.size > 0:
                    padding = np.full(window-1, errors_smoothed[0])
                    errors_smoothed = np.concatenate([padding, errors_smoothed])
                else: # If window > len(errors_np), just use the original small array
                    padding = np.full(window-1, errors_np[0] if errors_np.size > 0 else 0) # Use first element or 0
                    errors_smoothed = np.concatenate([padding, errors_np]) # Pad original errors

                reconstruction_error = torch.tensor(errors_smoothed, device=reconstruction_error.device, dtype=input_X.dtype)

            # Calculate threshold using quantile
            threshold = torch.quantile(reconstruction_error, quantile)
            
            # Generate anomaly labels
            anomaly_labels = (reconstruction_error > threshold).float()
            
            return reconstruction_error, anomaly_labels
    
    def get_topk_features(self,
                           input_X: torch.Tensor,
                           model_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the top k features for each instance in the input tensor.
        
        Args:
            input_X: Input tensor to analyze
            model_path: Path to the saved model
            
        Returns:
            Tuple containing:
            - per_feature_error: Reconstruction error per feature
            - active_latents: Indices of active latent units
            - latent_values: Values of active latent units
        """
        self.load_model(model_path)
        self.eval()
        
        with torch.no_grad():
            # Get reconstructed output and encoded features
            reconstructed_X, encoded_features, _ = self.forward_pass(input_X)
            
            # Calculate per-feature reconstruction error
            per_feature_error = (reconstructed_X - input_X) ** 2
            
            # Get indices of active latent units (where encoded_features > 0)
            active_latents = torch.nonzero(encoded_features, as_tuple=True)
            
            # Get values of active latent units
            latent_values = encoded_features[active_latents]
            
            return per_feature_error, active_latents, latent_values
            
    def analyze_anomaly(self,
                       input_X: torch.Tensor,
                       model_path: str,
                       top_n_features: int = 5) -> dict:
        """
        Analyze which input features contribute most to the anomaly detection.
        
        Args:
            input_X: Single input sample to analyze (shape: 1 x input_dim)
            model_path: Path to the saved model
            top_n_features: Number of top contributing features to return
            
        Returns:
            Dictionary containing:
            - top_feature_indices: Indices of features with highest reconstruction error
            - top_feature_errors: Reconstruction errors for top features
            - active_latents: Which latent units were activated
            - latent_values: Values of active latent units
            - total_reconstruction_error: Total reconstruction error for the sample
        """
        if input_X.dim() == 1:
            input_X = input_X.unsqueeze(0)
            
        per_feature_error, active_latents, latent_values = self.get_topk_features(input_X, model_path)
        
        # Get top N features with highest reconstruction error
        top_errors, top_indices = torch.topk(per_feature_error[0], k=top_n_features)
        
        # Calculate total reconstruction error
        total_error = per_feature_error.sum().item()
        
        return {
            'top_feature_indices': top_indices.tolist(),
            'top_feature_errors': top_errors.tolist(),
            'active_latents': active_latents[1].tolist(),  # Get column indices for batch size 1
            'latent_values': latent_values.tolist(),
            'total_reconstruction_error': total_error
        }


if __name__ == "__main__":
    train_dataset = torch.load("BATADAL_train_dataset.pt")
    val_dataset = torch.load("BATADAL_val_dataset.pt")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_dim = train_dataset.shape[1]
    print(f"Input dimension: {input_dim}")
    dict_size = 1024
    k = 32  # number of active latents to keep

    sae = TopKSAE(input_dim, dict_size, k)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=0.001)
    loss_history = sae.train_model(train_loader=train_loader, 
              valid_loader=val_loader, 
              optimizer=optimizer, 
              num_epochs=100)

    # Example of analyzing anomalies
    sample_idx = 0  # index of sample to analyze
    analysis = sae.analyze_anomaly(
        val_dataset[sample_idx],
        model_path="./checkpoints/2025-04-03_18-02-33/best_epoch.pth",
        top_n_features=5
    )
    print("\nAnomaly Analysis:")
    print(f"Top 5 contributing features: {analysis['top_feature_indices']}")
    print(f"Their reconstruction errors: {analysis['top_feature_errors']}")
    print(f"Active latent units: {analysis['active_latents']}")
    print(f"Latent values: {analysis['latent_values']}")
    print(f"Total reconstruction error: {analysis['total_reconstruction_error']}")
