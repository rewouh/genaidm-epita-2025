"""
This file implements the U-Net architecture used as the denoising network in DDPM.

The U-Net structure consists of:
- Encoder (downsampling path): progressively reduces spatial dimensions while increasing channels
- Bottleneck: processes features at the lowest resolution
- Decoder (upsampling path): progressively increases spatial dimensions while decreasing channels
- Skip connections: preserve fine-grained details from encoder to decoder

Key components:
- SinusoidalPositionEmbeddings: encodes timestep information using sinusoidal functions
- DownBlock/UpBlock: residual blocks with time embeddings and group normalization
- SimpleUNet: the complete U-Net architecture that predicts noise at each timestep

The model takes a noisy image and a timestep as input, and predicts the noise that was added.
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Paper Appendix B: "Diffusion time t is specified by adding the Transformer
    sinusoidal position embedding into each residual block."
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal position embeddings for timestep t.
        
        This creates a fixed-size representation of the timestep that can be
        added to the network's feature maps. The sinusoidal pattern allows the
        model to learn different behaviors at different noise levels.
        """
        device = time.device
        half_dim = self.dim // 2
        # Compute frequency divisors: creates different frequencies for each dimension
        embeddings = math.log(10000) / (half_dim - 1)
        # Generate frequency matrix: each dimension gets a different frequency
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Multiply timesteps with frequencies: creates phase-shifted patterns
        embeddings = time[:, None] * embeddings[None, :]
        # Concatenate sin and cos: provides both sine and cosine components for richer representation
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class DownBlock(nn.Module):
    """
    Paper Appendix B: "All models have two convolutional residual blocks per
    resolution level"
    
    Paper Appendix B: "We replaced weight normalization with group normalization
    to make the implementation simpler."
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        time_emb_proj = self.time_mlp(self.act(time_emb))[:, :, None, None]
        h = h + time_emb_proj
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.skip(x)


class UpBlock(nn.Module):
    """
    Upsampling block for the decoder path of the U-Net.
    
    Similar structure to DownBlock but used during upsampling to reconstruct
    the image from lower resolution features. Typically receives concatenated
    features from skip connections and upsampled features from the bottleneck.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        # First convolutional layer: processes concatenated features (skip + upsampled)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # Second convolutional layer: refines features
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # Group normalization layers
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        # SiLU activation
        self.act = nn.SiLU()
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # Skip connection for residual learning
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # First conv block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Inject timestep information
        time_emb_proj = self.time_mlp(self.act(time_emb))[:, :, None, None]
        h = h + time_emb_proj
        
        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        # Residual connection
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """
    Paper Section 4: "To represent the reverse process, we use a U-Net backbone
    similar to an unmasked PixelCNN++ with group normalization throughout."
    
    Paper Appendix B: "Our neural network architecture follows the backbone of
    PixelCNN++, which is a U-Net based on a Wide ResNet. We replaced weight
    normalization with group normalization to make the implementation simpler."
    
    Paper Appendix B: "Parameters are shared across time, which is specified to
    the network using the Transformer sinusoidal position embedding."
    """
    
    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.time_emb_dim = time_emb_dim
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Time embedding network: processes timestep into feature vector
        # Structure: sinusoidal embedding -> linear -> activation -> linear
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input convolution: initial feature extraction from noisy image
        self.conv_in = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        
        # Encoder path (downsampling): reduces spatial size, increases channels
        # First downsampling block: processes at full resolution
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)  # Halves spatial dimensions (28x28 -> 14x14)
        
        # Second downsampling block: processes at half resolution, doubles channels
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)  # Halves again (14x14 -> 7x7)
        
        # Bottleneck: processes features at lowest resolution (7x7)
        self.bottleneck = DownBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        
        # Decoder path (upsampling): increases spatial size, decreases channels
        # First upsampling: doubles spatial size (7x7 -> 14x14)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        # Processes concatenated features: upsampled (base*2) + skip (base*2) = base*4 channels
        self.up_block1 = UpBlock(base_channels * 4, base_channels, time_emb_dim)
        
        # Second upsampling: doubles spatial size (14x14 -> 28x28)
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        # Processes concatenated features: upsampled (base) + skip (base) = base*2 channels
        self.up_block2 = UpBlock(base_channels * 2, base_channels, time_emb_dim)
        
        # Output convolution: maps features back to image space (predicts noise)
        self.conv_out = nn.Conv2d(base_channels, image_channels, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net architecture.
        
        Paper Section 3.2: "We propose a specific parameterization motivated by [...] analysis"
        The model predicts the noise ε that was added to x_0 to get x_t.
        
        Paper Eq. (12): The objective "resembles denoising score matching over multiple
        noise scales indexed by t"
        
        Paper Section 3.2: "Optimizing an objective resembling denoising score matching
        is equivalent to using variational inference to fit the finite-time marginal of
        a sampling chain resembling Langevin dynamics."
        
        Args:
            x: Noisy image at timestep t, shape [batch, channels, height, width]
            t: Timestep indices, shape [batch]
        
        Returns:
            Predicted noise, same shape as input x
        """
        # Process timestep into embedding vector
        time_emb = self.time_mlp(t)
        
        # Encoder path: extract features at multiple resolutions
        # Initial feature extraction
        x1 = self.conv_in(x)
        x1 = self.dropout(x1)
        
        # First downsampling level: full resolution (28x28)
        x2 = self.down1(x1, time_emb)
        x2_pool = self.pool1(x2)  # Downsample to 14x14
        
        # Second downsampling level: half resolution (14x14)
        x3 = self.down2(x2_pool, time_emb)
        x3_pool = self.pool2(x3)  # Downsample to 7x7
        
        # Bottleneck: lowest resolution (7x7)
        x4 = self.bottleneck(x3_pool, time_emb)
        x4 = self.dropout(x4)
        
        # Decoder path: reconstruct image by upsampling and combining with skip connections
        # First upsampling: 7x7 -> 14x14
        x_up1 = self.up1(x4)
        # Concatenate with skip connection from encoder (preserves fine details)
        x_up1 = torch.cat([x_up1, x3], dim=1)
        x_up1 = self.up_block1(x_up1, time_emb)
        
        # Second upsampling: 14x14 -> 28x28
        x_up2 = self.up2(x_up1)
        # Concatenate with skip connection from first encoder level
        x_up2 = torch.cat([x_up2, x2], dim=1)
        x_up2 = self.up_block2(x_up2, time_emb)
        
        # Final output: predict noise (same shape as input)
        out = self.conv_out(x_up2)
        
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleUNet(image_channels=1, base_channels=64)
    logger.info(f"number of parameters: {count_parameters(model):,}")
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.randint(0, 1000, (batch_size,))
    
    out = model(x, t)
    logger.info(f"input shape: {x.shape}")
    logger.info(f"output shape: {out.shape}")
    assert out.shape == x.shape, "output must have the same size as input"
    logger.info("test passed")

