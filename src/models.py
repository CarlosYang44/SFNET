"""
SFNET - Model Architecture Module
Implements the multi-view Graph Autoencoder with heterogeneous attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv


# ============================================================================
# Attention Weight Configuration
# ============================================================================
# This weight controls the influence of the heterogeneous attention mechanism
# on the final cell and gene embeddings. Higher values give more weight to
# cross-view interactions, while lower values rely more on primary encodings.
HETERO_ATTENTION_WEIGHT = 0.1  # Increased from 0.1 for better cross-view fusion


class MultiviewEncoder(nn.Module):
    """
    Multi-view encoder that fuses cell-level and gene-level graph representations.
    
    This encoder processes two distinct graph views:
    1. Cell-level graph: Captures spatial and expression-based cell relationships
    2. Gene-level graph: Captures gene regulatory networks within and across cells
    
    The HeteroLayer enables cross-view attention to learn joint representations.
    
    Args:
        GeneEncoder: Encoder module for gene-level graph.
        CellEncoder: Encoder module for cell-level graph.
        heterolayer: Optional HeteroLayer for cross-view attention fusion.
        hidden_dim (int): Hidden dimension size. Default: 64
    """
    
    def __init__(self, GeneEncoder, CellEncoder, heterolayer=None, hidden_dim=64):
        super(MultiviewEncoder, self).__init__()
        self.encoder_gene = GeneEncoder
        self.encoder_cell = CellEncoder
        self.heterolayer = heterolayer
        self.hidden_dim = hidden_dim

    def forward(self, x_cell, x_gene, edge_index_cell, edge_index_gene):
        """
        Forward pass through the multi-view encoder.
        
        Args:
            x_cell (torch.Tensor): Cell node features.
            x_gene (torch.Tensor): Gene node features.
            edge_index_cell (torch.Tensor): Cell-level edge indices.
            edge_index_gene (torch.Tensor): Gene-level edge indices.
            
        Returns:
            tuple: (combined_embedding, cell_embedding, gene_embedding, raw_gene_embedding)
        """
        # Primary encoding for each view
        z_gene, gene_embeddings = self.encoder_gene(x_gene, edge_index_gene)
        z_cell = self.encoder_cell(x_cell, edge_index_cell)

        # Apply heterogeneous attention for cross-view fusion
        if self.heterolayer is not None:
            feature_dict = {'cell': z_cell, 'gene': z_gene}
            hetero_output = self.heterolayer(feature_dict)

            # Fuse heterolayer output with primary embeddings
            # Using configurable weight for attention contribution
            z_cell = z_cell + hetero_output['cell'] * HETERO_ATTENTION_WEIGHT
            z_gene = z_gene + hetero_output['gene'] * HETERO_ATTENTION_WEIGHT

        # Concatenate embeddings from both views
        z_combined = torch.cat((z_cell, z_gene), dim=1)
        
        return z_combined, z_cell, z_gene, gene_embeddings


class HeteroLayer(nn.Module):
    """
    Heterogeneous attention layer for cross-view feature interaction.
    
    This layer applies multi-head self-attention to jointly model cell and gene
    features, enabling information flow between the two graph views.
    
    Args:
        hidden_dim (int): Feature dimension for attention computation.
        num_heads (int): Number of attention heads. Default: 4
        dropout (float): Dropout rate for attention weights. Default: 0.1
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(HeteroLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict):
        """
        Process cell and gene features with cross-view attention.
        
        Args:
            x_dict (dict): Dictionary with 'cell' and 'gene' feature tensors.
            
        Returns:
            dict: Attended features for 'cell' and 'gene'.
        """
        cell_features = x_dict['cell']
        gene_features = x_dict['gene']
        
        # Concatenate cell and gene features for joint attention
        combined_features = torch.cat([cell_features, gene_features], dim=0)
        
        # Apply multi-head self-attention
        attended_features, attention_weights = self.attention(
            combined_features, 
            combined_features, 
            combined_features
        )
        
        # Apply residual connection with layer normalization
        attended_features = self.layer_norm(combined_features + self.dropout(attended_features))
        
        # Split attended features back to cell and gene components
        num_cells = cell_features.size(0)
        cell_output = attended_features[:num_cells]
        gene_output = attended_features[num_cells:]
        
        return {'cell': cell_output, 'gene': gene_output}


class CellEncoder(torch.nn.Module):
    """
    Graph Convolutional Network encoder for cell-level graph.
    
    Encodes cell nodes based on their features and spatial/expression-based
    neighborhood relationships.
    
    Args:
        num_features (int): Input feature dimension.
        hidden_dim (int): Output embedding dimension.
        dropout (float): Dropout rate. Default: 0.2
        is_training (bool): Training mode flag. Default: False
    """
    
    def __init__(self, num_features, hidden_dim, dropout=0.2, is_training=False):
        super(CellEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.is_training = is_training
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass through the cell encoder.
        
        Args:
            x (torch.Tensor): Cell node features.
            edge_index (torch.Tensor): Edge indices for cell graph.
            
        Returns:
            torch.Tensor: Cell node embeddings.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GeneEncoder(torch.nn.Module):
    """
    Graph Convolutional Network encoder for gene-level graph.
    
    Encodes gene nodes and aggregates them to cell-level representations.
    Each cell has multiple gene nodes, which are aggregated via a linear layer.
    
    Args:
        num_features (int): Input feature dimension.
        hidden_dim (int): Output embedding dimension.
        num_vertices (int): Number of cells (vertices in cell graph).
        num_subvertices (int): Number of genes per cell.
        dropout (float): Dropout rate. Default: 0.2
        is_training (bool): Training mode flag. Default: False
    """
    
    def __init__(self, num_features, hidden_dim, num_vertices, num_subvertices, 
                 dropout=0.2, is_training=False):
        super(GeneEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.aggregation_layer = nn.Linear(num_subvertices * hidden_dim, hidden_dim)
        self.is_training = is_training
        self.dropout = dropout
        self.num_vertices = num_vertices
        self.num_subvertices = num_subvertices

    def embed(self, x, edge_index):
        """
        Compute gene-level embeddings.
        
        Args:
            x (torch.Tensor): Gene node features.
            edge_index (torch.Tensor): Edge indices for gene graph.
            
        Returns:
            torch.Tensor: Gene node embeddings.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        """
        Forward pass with gene-to-cell aggregation.
        
        Args:
            x (torch.Tensor): Gene node features.
            edge_index (torch.Tensor): Edge indices for gene graph.
            
        Returns:
            tuple: (cell_level_aggregated_embedding, raw_gene_embeddings)
        """
        embeddings = self.embed(x, edge_index)
        
        # Reshape and aggregate genes to cell level
        # Shape: (num_cells * num_genes, hidden_dim) -> (num_cells, num_genes * hidden_dim)
        x = embeddings.view(
            embeddings.shape[0] // self.num_subvertices, 
            embeddings.shape[1] * self.num_subvertices
        )
        x = self.aggregation_layer(x)
        
        return x, embeddings


class InnerProductDecoder(torch.nn.Module):
    """
    Inner product decoder for link prediction.
    
    Reconstructs the adjacency matrix by computing inner products between
    node embeddings.
    """
    
    def forward(self, z, edge_index, sigmoid=True):
        """
        Decode edge probabilities from node embeddings.
        
        Args:
            z (torch.Tensor): Node embeddings.
            edge_index (torch.Tensor): Edge indices to decode.
            sigmoid (bool): Whether to apply sigmoid activation. Default: True
            
        Returns:
            torch.Tensor: Edge probability scores.
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value
