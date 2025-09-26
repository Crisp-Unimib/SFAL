import torch
import torch.nn as nn

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

class TopKSAE(nn.Module):
    """
    Defines the Top-K Sparse Autoencoder architecture.
    For each input token, it activates the 'k' features with the highest
    pre-activation scores.
    """
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        """
        Encodes input activations into sparse feature activations.
        """
        # Calculate pre-activations
        pre_acts = input_acts @ self.W_enc + self.b_enc
        
        # Find the top k pre-activations along the feature dimension
        top_k_values, top_k_indices = torch.topk(pre_acts, self.k, dim=-1)
        
        # Apply ReLU to only the top-k values
        relu_top_k_values = torch.relu(top_k_values)
        
        # Create a sparse tensor for the final activations
        acts = torch.zeros_like(pre_acts)
        
        # Place the positive top-k values back into the tensor
        acts.scatter_(-1, top_k_indices, relu_top_k_values)
        
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon
