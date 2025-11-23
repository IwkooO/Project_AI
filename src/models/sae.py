import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder (SAE).
    Encodes input embeddings into a sparse combination of dictionary elements.
    Sparsity is enforced by keeping only the top-k activations.
    """
    def __init__(self, input_dim: int, expansion_factor: int = 8, k: int = 32):
        """
        Args:
            input_dim: Dimension of input embedding (e.g., 512 for StreetCLIP).
            expansion_factor: Ratio of dictionary size to input dim.
            k: Number of active neurons (Top-K sparsity).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        self.k = k
        
        # Encoder (Expansion)
        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        # Initialize encoder bias to zero
        self.encoder.bias.data.zero_()
        
        # Decoder (Reconstruction)
        # Note: We often constrain decoder columns to have unit norm
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=False)
        
        # Tie weights? Usually NO for SAEs, we let them diverge.
        # But we should initialize decoder with unit norm columns.
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, input_dim)
            
        Returns:
            reconstructed: (B, input_dim)
            acts: Sparse activations (B, hidden_dim)
            loss: Scalar reconstruction loss (MSE)
        """
        # 1. Encode
        # Pre-activation
        pre_acts = self.encoder(x)
        
        # 2. Top-K Sparsity
        # Keep only the top k values, zero out the rest
        topk_values, topk_indices = torch.topk(pre_acts, k=self.k, dim=-1)
        
        # Create sparse mask
        mask = torch.zeros_like(pre_acts)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # Apply activation (ReLU) and Mask
        # Note: Some Top-K implementations use ReLU before Top-K, some after.
        # Usually ReLU(x) -> TopK is better to ignore negative correlations if desired.
        # Here: standard Top-K SAE usually does ReLU(pre_acts) * mask
        
        acts = F.relu(pre_acts) * mask
        
        # 3. Decode
        reconstructed = self.decoder(acts)
        
        # 4. Loss
        # MSE Reconstruction Loss
        loss = F.mse_loss(reconstructed, x)
        
        return reconstructed, acts, loss
        
    @torch.no_grad()
    def normalize_decoder(self):
        """
        Enforce unit norm constraint on decoder weight columns.
        Should be called after every optimization step.
        """
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, p=2, dim=0)
        
    @torch.no_grad()
    def remove_parallel_component(self, grad, weight):
        """
        Remove the component of the gradient parallel to the decoder weights.
        Used if we want strict unit norm optimization, but simple normalization
        after step is usually sufficient and more stable.
        """
        # Not strictly needed if using normalize_decoder()
        pass

    @torch.no_grad()
    def get_dead_neurons(self, activation_counts, threshold=0):
        """
        Identify neurons that haven't fired.
        Args:
            activation_counts: Tensor of shape (hidden_dim,) counting activations.
            threshold: Minimum count to be considered "alive".
        """
        return (activation_counts <= threshold)

    @torch.no_grad()
    def reset_dead_neurons(self, dead_mask, data_stats):
        """
        Resample dead neurons.
        Args:
            dead_mask: Boolean mask of dead neurons (hidden_dim,)
            data_stats: Dictionary containing 'mean' and 'std' of input data or a batch of input data.
        """
        # This needs a batch of current data to resample towards.
        # Implemented in training loop.
        pass

