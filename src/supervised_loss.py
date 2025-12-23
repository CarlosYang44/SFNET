"""
SFNET - Contrastive Loss Module
Implements contrastive learning loss for self-supervised representation learning.
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for self-supervised learning.
    
    This loss encourages similar samples to have similar representations
    while pushing dissimilar samples apart in the embedding space.
    
    Args:
        batch_size (int): The batch size for computing the loss.
        temperature (float): Temperature parameter for scaling the similarity scores.
            Lower values make the distribution sharper. Default: 0.5
    """
    
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        # Dynamically initialized negatives mask (lazy initialization)
        self.negatives_mask = None

    def forward(self, emb_i, emb_j):
        """
        Compute the contrastive loss between two sets of embeddings.
        
        Args:
            emb_i (torch.Tensor): First set of embeddings, shape (batch_size, feature_dim)
            emb_j (torch.Tensor): Second set of embeddings, shape (batch_size, feature_dim)
            
        Returns:
            torch.Tensor: Scalar contrastive loss value
        """
        # Normalize embeddings to unit vectors
        z_i = F.normalize(emb_i, dim=1)     
        z_j = F.normalize(emb_j, dim=1)

        # Concatenate embeddings from both views
        representations = torch.cat([z_i, z_j], dim=0)  # Shape: (2 * batch_size, feature_dim)
        
        # Compute pairwise cosine similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), 
            representations.unsqueeze(0), 
            dim=2
        )  # Shape: (2 * batch_size, 2 * batch_size)

        # Dynamically initialize negatives mask if needed
        if self.negatives_mask is None or self.negatives_mask.size(0) != similarity_matrix.size(0):
            self.negatives_mask = (
                ~torch.eye(similarity_matrix.size(0), dtype=torch.bool)
            ).float().to(similarity_matrix.device)

        # Extract positive pairs at different diagonal offsets
        sim_ij_bs1 = torch.diag(similarity_matrix, self.batch_size)
        sim_ji_bs1 = torch.diag(similarity_matrix, -self.batch_size)
        sim_ij_bs2 = torch.diag(similarity_matrix, 2 * self.batch_size)
        sim_ji_bs2 = torch.diag(similarity_matrix, -2 * self.batch_size)
        sim_ij_bs3 = torch.diag(similarity_matrix, 3 * self.batch_size)
        sim_ji_bs3 = torch.diag(similarity_matrix, -3 * self.batch_size)

        # Combine positive pairs from all offsets
        positives1 = torch.cat([sim_ij_bs1, sim_ji_bs1], dim=0)
        positives2 = torch.cat([sim_ij_bs2, sim_ji_bs2], dim=0)
        positives3 = torch.cat([sim_ij_bs3, sim_ji_bs3], dim=0)

        # Compute numerator: exp(positive_similarity / temperature)
        nominator1 = torch.exp(positives1 / self.temperature)
        nominator2 = torch.exp(positives2 / self.temperature)
        nominator3 = torch.exp(positives3 / self.temperature)
        nominator = torch.sum(nominator1) + torch.sum(nominator2) + torch.sum(nominator3)
        
        # Compute denominator: sum of all negative pair similarities
        denominator = (
            torch.sum(self.negatives_mask * torch.exp(similarity_matrix / self.temperature)) 
            - nominator
        )

        # Compute InfoNCE-style contrastive loss
        loss_partial = -torch.log(nominator / denominator)
        loss = torch.sum(loss_partial) / (4 * self.batch_size)
        
        return loss

    
if __name__ == '__main__':
    # Unit test for ContrastiveLoss
    dummy_x = torch.randn([1597, 64]).float().cpu()
    dummy_y = torch.randn([1597, 64]).float().cpu()
    model = ContrastiveLoss(800 * 3).cpu()
    output = model(dummy_x, dummy_y)
    print(f"Contrastive Loss Output: {output}")
