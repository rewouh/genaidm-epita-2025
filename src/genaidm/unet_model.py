import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class DownBlock(nn.Module):
    
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


class SimpleUNet(nn.Module):
    
    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 128
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.time_emb_dim = time_emb_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.conv_in = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = DownBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.up_block1 = UpBlock(base_channels * 4, base_channels, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.up_block2 = UpBlock(base_channels * 2, base_channels, time_emb_dim)
        
        self.conv_out = nn.Conv2d(base_channels, image_channels, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        
        x1 = self.conv_in(x)
        
        x2 = self.down1(x1, time_emb)
        x2_pool = self.pool1(x2)
        
        x3 = self.down2(x2_pool, time_emb)
        x3_pool = self.pool2(x3)
        
        x4 = self.bottleneck(x3_pool, time_emb)
        
        x_up1 = self.up1(x4)
        x_up1 = torch.cat([x_up1, x3], dim=1)
        x_up1 = self.up_block1(x_up1, time_emb)
        
        x_up2 = self.up2(x_up1)
        x_up2 = torch.cat([x_up2, x2], dim=1)
        x_up2 = self.up_block2(x_up2, time_emb)
        
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

