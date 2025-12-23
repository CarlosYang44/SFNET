"""
SFNET - Spatial Enhancement Block (SEB)
Implements multi-scale convolution with selective kernel attention for spatial features,
combined with Graph Attention Networks for global context.
"""
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class SelectiveKernelBlock(nn.Module):
    """
    Selective Kernel Block for multi-scale feature extraction.
    
    Applies convolutions with different kernel sizes and adaptively
    selects features through attention-based fusion.
    
    Reference: Selective Kernel Networks (SKNet)
    
    Args:
        channel (int): Number of input/output channels.
        kernels (list): List of kernel sizes. Default: [1, 3, 5, 7]
        reduction (int): Channel reduction ratio for attention. Default: 16
        group (int): Number of groups for grouped convolution. Default: 1
        min_channels (int): Minimum intermediate channels. Default: 32
    """
    
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, min_channels=32):
        super(SelectiveKernelBlock, self).__init__()
        
        # Minimum intermediate dimension
        self.d = max(min_channels, channel // reduction)
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        
        # Attention computation
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        Forward pass with selective kernel attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (N, C, H, W).
        """
        batch_size, channels, _, _ = x.size()
        
        # Apply all kernel convolutions
        conv_outs = [conv(x) for conv in self.convs]
        feats = torch.stack(conv_outs, dim=0)  # Shape: (num_kernels, N, C, H, W)

        # Fuse features
        fused = sum(conv_outs)

        # Global average pooling for attention
        global_avg = fused.mean(-1).mean(-1)  # Shape: (N, C)
        z = self.fc(global_avg)  # Shape: (N, d)

        # Compute attention weights for each kernel
        weights = []
        for fc in self.fcs:
            weight = fc(z)  # Shape: (N, C)
            weights.append(weight.view(batch_size, channels, 1, 1))
        
        attention_weights = torch.stack(weights, dim=0)  # Shape: (num_kernels, N, C, 1, 1)
        attention_weights = self.softmax(attention_weights)

        # Apply attention and aggregate
        output = (attention_weights * feats).sum(dim=0)
        
        return output


class LocalEnhancementBlock(nn.Module):
    """
    Local Enhancement Block for 1D feature tensors.
    
    Wraps SelectiveKernelBlock to work with 1D embeddings.
    
    Args:
        in_channels (int): Input feature dimension.
    """
    
    def __init__(self, in_channels):
        super(LocalEnhancementBlock, self).__init__()
        self.selective_kernel = SelectiveKernelBlock(
            channel=in_channels, 
            reduction=in_channels / 4
        )

    def forward(self, x):
        """
        Forward pass for 1D embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C).
            
        Returns:
            torch.Tensor: Enhanced tensor of shape (N, C).
        """
        # Ensure 2D input
        x = x.view([x.shape[0], x.shape[1]])
        batch_size, channels = x.shape
        
        # Reshape to 4D for selective kernel
        x = x.view(batch_size, channels, 1, 1)
        
        # Apply selective kernel attention
        x_enhanced = self.selective_kernel(x)
        
        # Reshape back to 2D
        x_enhanced = x_enhanced.view(x_enhanced.shape[0], x_enhanced.shape[1])
        
        return x_enhanced


class GlobalEnhancementBlock(torch.nn.Module):
    """
    Global Enhancement Block using Graph Attention Networks.
    
    Combines GAT layers with local enhancement for both global graph
    structure and local feature refinement.
    
    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.
        heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.4
        bias (bool): Whether to use bias. Default: True
    """
    
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.4, bias=True):
        super(GlobalEnhancementBlock, self).__init__()
        
        # First GAT layer with multi-head attention (concatenated output)
        self.conv1 = GATConv(
            in_channels, 
            out_channels, 
            heads=heads, 
            concat=True, 
            dropout=dropout, 
            bias=bias
        )
        
        # Local enhancement for intermediate features
        self.local_enhancement = LocalEnhancementBlock(heads * out_channels)
        
        # Second GAT layer with averaged multi-head output
        self.conv2 = GATConv(
            heads * out_channels, 
            out_channels, 
            heads=heads, 
            concat=False, 
            dropout=dropout, 
            bias=bias
        )

    def forward(self, data):
        """
        Forward pass through global enhancement.
        
        Args:
            data (Data): PyTorch Geometric Data object with x and edge_index.
            
        Returns:
            tuple: (output_features, intermediate_features)
        """
        x, edge_index = data.x, data.edge_index
        
        # First GAT layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        
        # Local enhancement
        x = self.local_enhancement(x)
        intermediate_features = x  # Store for skip connection
        
        # Second GAT layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1), intermediate_features


# Legacy aliases for backward compatibility
Local_stage1 = SelectiveKernelBlock
Local_stage2 = LocalEnhancementBlock
Global = GlobalEnhancementBlock


if __name__ == '__main__':
    # Unit test
    print("Testing SEB Module...")
    
    dummy_input = torch.rand([100, 32])
    dummy_edge = torch.randint(0, 100, [2, 10000])
    data = Data(x=dummy_input, edge_index=dummy_edge)
    
    in_channels = 32
    out_channels = 64
    
    seb_model = GlobalEnhancementBlock(in_channels, out_channels)
    output, intermediate = seb_model(data)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Intermediate shape: {intermediate.shape}")
    print("Test passed!")
