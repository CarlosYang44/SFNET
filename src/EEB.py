"""
SFNET - Embedding Enhancement Block (EEB)
Implements coordinate attention mechanism for feature enhancement.

Reference: Coordinate Attention for Efficient Mobile Network Design
"""
import torch
import torch.nn as nn


class HardSigmoid(nn.Module):
    """
    Hard Sigmoid activation function.
    
    A computationally efficient approximation of the sigmoid function:
    HardSigmoid(x) = ReLU6(x + 3) / 6
    
    Args:
        inplace (bool): If True, modify tensor in-place. Default: True
    """
    
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HardSwish(nn.Module):
    """
    Hard Swish activation function.
    
    A computationally efficient approximation of the Swish function:
    HardSwish(x) = x * HardSigmoid(x)
    
    Args:
        inplace (bool): If True, modify tensor in-place. Default: True
    """
    
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention block for spatial feature enhancement.
    
    This block captures long-range dependencies with positional information
    by decomposing channel attention into two 1D feature encoding processes.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        reduction (int): Channel reduction ratio. Default: 32
    """
    
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        
        # Adaptive pooling for each direction
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Intermediate channels
        mid_channels = max(8, in_channels // reduction)

        # Shared convolution for dimension reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = HardSwish()

        # Separate convolutions for height and width attention
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass through coordinate attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Attention-enhanced tensor of same shape.
        """
        identity = x
        n, c, h, w = x.size()
        
        # Pool along each spatial dimension
        x_h = self.pool_h(x)  # Shape: (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # Shape: (N, C, W, 1) -> (N, C, 1, W) transposed

        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back to height and width components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention
        out = identity * a_w * a_h

        return out


class EmbeddingEnhancementBlock(nn.Module):
    """
    Embedding Enhancement Block (EEB) for 1D feature tensors.
    
    Wraps CoordinateAttention to work with 1D embeddings by reshaping
    to 4D, applying attention, and reshaping back.
    
    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.
    """
    
    def __init__(self, in_channels, out_channels):
        super(EmbeddingEnhancementBlock, self).__init__()
        self.coord_attention = CoordinateAttention(
            in_channels, 
            out_channels, 
            reduction=in_channels / 4
        )

    def forward(self, x):
        """
        Forward pass for 1D embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C).
            
        Returns:
            torch.Tensor: Enhanced tensor of shape (N, out_channels).
        """
        batch_size, channels = x.shape
        
        # Reshape to 4D for coordinate attention
        x = x.view(batch_size, channels, 1, 1)
        
        # Apply coordinate attention
        x_enhanced = self.coord_attention(x)
        
        # Reshape back to 2D
        x_enhanced = x_enhanced.view(x_enhanced.shape[0], x_enhanced.shape[1])
        
        return x_enhanced


# Legacy aliases for backward compatibility
EEB_stage1 = CoordinateAttention
EEB_stage2 = EmbeddingEnhancementBlock
h_sigmoid = HardSigmoid
h_swish = HardSwish


if __name__ == '__main__':
    # Unit test
    print("Testing EEB Module...")
    dummy_input = torch.rand([100, 128])
    in_channels = 128
    out_channels = 128
    
    eeb_model = EmbeddingEnhancementBlock(in_channels=in_channels, out_channels=out_channels)
    output = eeb_model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Shape mismatch!"
    print("Test passed!")
