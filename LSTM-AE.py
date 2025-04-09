import torch
import torch.nn as nn
import os
import numpy as np
import datetime

class LSTMAutoEncoder(nn.Module):
    """
    An LSTM-based autoencoder for temporal data.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final linear layer to map hidden state to output
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, 
               input_X: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input sequence to latent space.
        Args:
            input_X: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tuple of:
            - Encoded sequence of shape (batch_size, seq_len, hidden_dim)
            - Tuple of (hidden_state, cell_state) for the encoder
        """
        output, (hidden, cell) = self.encoder(input_X)
        return output, (hidden, cell)
    
    def decode(self, 
               encoded_seq: torch.Tensor,
               encoder_states: tuple[torch.Tensor, torch.Tensor],
               target_len: int) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        Args:
            encoded_seq: Last step of encoded sequence of shape (batch_size, 1, hidden_dim)
            encoder_states: Tuple of (hidden_state, cell_state) from encoder
            target_len: Length of sequence to generate
        Returns:
            Reconstructed sequence of shape (batch_size, target_len, input_dim)
        """
        # Initialize decoder input as the last encoded state
        decoder_input = encoded_seq[:, -1:, :]  # Shape: (batch_size, 1, hidden_dim)
        
        # Use encoder final states to initialize decoder states
        decoder_states = encoder_states
        
        outputs = []
        for _ in range(target_len):
            # Run decoder for one step
            output, decoder_states = self.decoder(decoder_input, decoder_states)
            
            # Map to input space
            output = self.output_layer(output)
            outputs.append(output)
            
            # Use current output as next input
            decoder_input = output
            
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)
    
    def forward_pass(self, 
                     input_X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Args:
            input_X: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Tuple of (reconstructed_sequence, encoded_sequence, reconstruction_loss)
        """
        # Encode input sequence
        encoded_seq, encoder_states = self.encode(input_X)
        
        # Decode from the encoded sequence
        reconstructed_seq = self.decode(
            encoded_seq,
            encoder_states,
            target_len=input_X.size(1)
        )
        
        # Compute reconstruction loss using MSE
        recon_loss = torch.mean(torch.sum((reconstructed_seq - input_X) ** 2, dim=(1, 2)))
        
        return reconstructed_seq, encoded_seq, recon_loss

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
                reconstructed_seq, encoded_seq, train_loss = self.forward_pass(input_X)
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
                    reconstructed_seq, encoded_seq, valid_loss = self.forward_pass(input_X)
                    valid_losses.append(valid_loss.item())

            mean_valid_loss = np.mean(valid_losses)
            print(f"Epoch {epoch}, Valid Loss: {mean_valid_loss}")
            loss_history["valid"].append(mean_valid_loss)

            if mean_valid_loss < best_valid_loss:
                best_valid_loss = mean_valid_loss
                best_epoch = epoch
                save_path = f"./checkpoints/{date_time}"
                os.makedirs(save_path, exist_ok=True)
                torch.save(self.state_dict(), f"{save_path}/best_epoch.pth")
                print(f"Saved model with lowest valid loss: {best_valid_loss} at epoch {best_epoch}")

        return loss_history

    def load_model(self,
                   model_path: str):
        """Load a pretrained model."""
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
    
    def predict_anomaly(self,
                        model_path: str,
                        input_X: torch.Tensor,
                        quantile: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomalies using reconstruction error and quantile threshold.
        Args:
            model_path: Path to the saved model file
            input_X: Input tensor of shape (batch_size, seq_len, input_dim)
            quantile: Quantile threshold for anomaly detection (default: 0.95)
        Returns:
            Tuple containing:
            - Reconstruction error for each sequence
            - Binary labels indicating anomalies (1) and normal samples (0)
        """
        self.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        self.eval()
        with torch.no_grad():
            reconstructed_seq, encoded_seq, _ = self.forward_pass(input_X)
            # Use MSE as reconstruction error
            reconstruction_error = torch.sum((reconstructed_seq - input_X) ** 2, dim=(1, 2))
            
            # Calculate threshold using quantile
            threshold = torch.quantile(reconstruction_error, quantile)
            
            # Generate anomaly labels
            anomaly_labels = (reconstruction_error > threshold).float()
            
            return reconstruction_error, anomaly_labels

def safe_load_tensor(file_path: str) -> torch.Tensor:
    """
    Safely load a tensor from a file using weights_only=True.
    
    Args:
        file_path: Path to the tensor file
    Returns:
        Loaded tensor
    """
    try:
        # Load tensor with weights_only=True for security
        tensor = torch.load(file_path, weights_only=True)
        return tensor
    except Exception as e:
        print(f"Error loading tensor from {file_path}: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    input_dim = 43  # Number of features in BATADAL dataset
    hidden_dim = 32  # Size of latent space
    num_layers = 2  # Number of LSTM layers
    
    # Create model
    lstm_ae = LSTMAutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(lstm_ae.parameters(), lr=0.001)
    
    # Safely load the datasets
    try:
        train_dataset = safe_load_tensor("BATADAL_train_dataset.pt")
        val_dataset = safe_load_tensor("BATADAL_val_dataset.pt")
        print(f"Successfully loaded datasets:")
        print(f"Train dataset shape: {train_dataset.shape}")
        print(f"Validation dataset shape: {val_dataset.shape}")
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        raise
    
    # Note: For BATADAL dataset, you'll need to reshape your data into sequences
    # Example: reshape (N, F) into (batch_size, seq_len, input_dim)
    # where N is total samples and F is number of features
    
    # Example of how to prepare the data (pseudo-code):
    # seq_len = 24  # e.g., 24 hours as one sequence
    # train_sequences = create_sequences(train_data, seq_len)
    # train_dataset = TensorDataset(train_sequences)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # No shuffle for temporal data
    
    # Train the model
    # loss_history = lstm_ae.train_model(train_loader, valid_loader, optimizer, num_epochs=100)
