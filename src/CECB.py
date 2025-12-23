"""
SFNET - Channel Enhancement and Calibration Block (CECB)
Implements channel-wise feature calibration using batch normalization
and squeeze-excitation-style attention.
"""
import torch
import torch.nn as nn
import random


class ChannelEnhancementBlock(torch.nn.Module):
    """
    Channel Enhancement and Calibration Block (CECB).
    
    This block enhances feature representations through:
    1. Batch normalization to compute mean statistics
    2. Residual extraction (input - normalized)
    3. Squeeze-Excitation-style channel attention on residuals
    4. Recombination of normalized and calibrated residual features
    
    Inspired by SENet and batch normalization calibration techniques.
    
    Args:
        in_channels (int): Number of input channels/features.
        intermediate_channels (int): Hidden dimension in SE block. Default: 32
    """
    
    def __init__(self, in_channels, intermediate_channels=32):
        super(ChannelEnhancementBlock, self).__init__()
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Squeeze-Excitation layers for residual calibration
        self.fc1 = nn.Linear(in_channels, intermediate_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_channels, in_channels)
        self.sigmoid = nn.Sigmoid()
        
        # Batch normalization for mean statistics
        self.bn = nn.BatchNorm2d(in_channels)
        
        # Final pooling
        self.gap_final = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward pass through channel enhancement block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C) or (N, C, 1, 1).
            
        Returns:
            torch.Tensor: Enhanced tensor of shape (N, C).
        """
        # Ensure 4D input for batch norm
        x = x.view([x.shape[0], x.shape[1], 1, 1])
        
        # Compute normalized features (mean statistics)
        x_normalized = self.bn(x)
        
        # Extract residual (deviation from mean)
        x_residual = x - x_normalized
        
        # Squeeze: Global average pooling on residual
        x_squeezed = self.gap(x_residual)
        x_squeezed = x_squeezed.view([x.shape[0], x.shape[1]])
        
        # Excitation: Channel attention weights
        x_excite = self.fc1(x_squeezed)
        x_excite = self.relu(x_excite)
        x_excite = self.fc2(x_excite)
        x_excite = self.sigmoid(x_excite)
        x_excite = x_excite.view([x.shape[0], x.shape[1], 1, 1])
        
        # Scale residual with attention
        x_calibrated = torch.mul(x_residual, x_excite)
        
        # Combine normalized and calibrated residual
        x_combined = x_normalized + x_calibrated
        
        # Final normalization and pooling
        x_output = self.bn(x_combined)
        x_output = self.gap_final(x_output)
        
        # Reshape to 2D output
        x_output = x_output.view(x_output.shape[0], x_output.shape[1])
        
        return x_output


# Legacy alias for backward compatibility
CECB = ChannelEnhancementBlock


if __name__ == "__main__":
    # Unit test
    print("Testing CECB Module...")
    
    # Test with cell-level feature dimension
    dummy_cell = torch.rand([1597, 64])
    model_cell = ChannelEnhancementBlock(64)
    out_cell = model_cell(dummy_cell)
    print(f"Cell input shape: {dummy_cell.shape}")
    print(f"Cell output shape: {out_cell.shape}")
    assert out_cell.shape == dummy_cell.shape, "Cell shape mismatch!"
    
    # Test with gene-level feature dimension
    dummy_gene = torch.rand([71865, 32])
    model_gene = ChannelEnhancementBlock(32)
    out_gene = model_gene(dummy_gene)
    print(f"Gene input shape: {dummy_gene.shape}")
    print(f"Gene output shape: {out_gene.shape}")
    assert out_gene.shape == dummy_gene.shape, "Gene shape mismatch!"
    
    print("All tests passed!")
