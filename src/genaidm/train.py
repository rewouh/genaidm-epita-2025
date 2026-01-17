
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)
from typing import Optional

from genaidm.ddpm_mnist import DDPM
from genaidm.unet_model import SimpleUNet


def train_ddpm(
    model: nn.Module,
    ddpm: DDPM,
    dataloader: DataLoader,
    num_epochs: int,
    device: str,
    lr: float = 2e-4,
    save_dir: str = "checkpoints",
    save_every: int = 10,
    use_ema: bool = True,
    ema_decay: float = 0.9999
) -> list:
    """
    Paper Section 3.4: "We found it beneficial to sample quality (and simpler
    to implement) to train on [...] simplified objective" (Eq. 14)
    
    Paper Appendix B: "We set the learning rate to 2 × 10⁻⁴ without any sweeping"
    Paper Appendix B: "We used EMA on model parameters with a decay factor of 0.9999"
    """
    model.to(device)
    # Paper Appendix B: "We tried Adam and RMSProp early on [...] and chose the former"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    if use_ema:
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs * len(dataloader)
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    losses = []
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Paper Section 3.3: "We assume that image data consists of integers in
            # {0, 1, ..., 255} scaled linearly to [−1, 1]. This ensures that the neural
            # network reverse process operates on consistently scaled inputs starting
            # from the standard normal prior p(x_T)."
            images = images * 2 - 1
            
            loss = ddpm.compute_loss(model, images, device)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            if use_ema:
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
            
            loss_val = loss.item()
            losses.append(loss_val)
            epoch_losses.append(loss_val)
            
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'avg_loss': f'{sum(epoch_losses)/len(epoch_losses):.4f}'
            })
            
            global_step += 1
        
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"ddpm_epoch_{epoch+1}.pt")
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': sum(epoch_losses) / len(epoch_losses),
            }
            if use_ema:
                save_dict['ema_model_state_dict'] = ema_model.state_dict()
            torch.save(save_dict, checkpoint_path)
            logger.info(f"checkpoint saved: {checkpoint_path}")
    
    final_path = os.path.join(save_dir, "ddpm_final.pt")
    save_dict = {
        'model_state_dict': ema_model.state_dict() if use_ema else model.state_dict(),
        'ddpm_config': {
            'num_timesteps': ddpm.num_timesteps,
            'beta_start': ddpm.betas[0].item(),
            'beta_end': ddpm.betas[-1].item()
        }
    }
    torch.save(save_dict, final_path)
    logger.info(f"final model saved: {final_path}")
    logger.info(f"using {'EMA' if use_ema else 'standard'} weights for final model")
    
    # Paper Section 4.3: "The gap between train and test is at most 0.03 bits per
    # dimension, which is comparable to the gaps reported with other likelihood-based
    # models and indicates that our diffusion model is not overfitting"
    
    return losses


def get_mnist_dataloader(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    use_flip: bool = False
) -> DataLoader:
    """Get MNIST dataloader with optional data augmentation.
    
    Note: Horizontal flips disabled by default for MNIST since flipping
    creates invalid digits (6↔9, etc.). The paper used flips for natural images.
    """
    transforms_list = [transforms.ToTensor()]
    if use_flip:
        transforms_list.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    
    transform = transforms.Compose(transforms_list)
    
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    ddpm: DDPM,
    num_samples: int,
    device: str,
    return_trajectory: bool = False
) -> torch.Tensor:
    """
    Paper Section 4.3 (Progressive generation): "We also run a progressive
    unconditional generation process given by progressive decompression from
    random bits. [...] Large scale image features appear first and details
    appear last."
    """
    model.eval()
    shape = (num_samples, 1, 28, 28)
    
    samples = ddpm.p_sample_loop(
        model,
        shape,
        device,
        return_trajectory=return_trajectory
    )
    
    if return_trajectory:
        samples = (samples + 1) / 2
    else:
        samples = (samples + 1) / 2
    
    return samples


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: str
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    logger.info(f"checkpoint loaded: {checkpoint_path}")
    
    if 'epoch' in checkpoint:
        logger.info(f"epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        logger.info(f"loss: {checkpoint['loss']:.4f}")
    
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")
    
    num_timesteps = 1000
    num_epochs = 20
    batch_size = 128
    lr = 2e-4
    
    model = SimpleUNet(image_channels=1, base_channels=64)
    ddpm = DDPM(num_timesteps=num_timesteps, beta_start=1e-4, beta_end=0.02)
    
    logger.info(f"number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dataloader = get_mnist_dataloader(batch_size=batch_size)
    logger.info(f"number of batches per epoch: {len(dataloader)}")
    
    logger.info("starting training")
    losses = train_ddpm(
        model=model,
        ddpm=ddpm,
        dataloader=dataloader,
        num_epochs=num_epochs,
        device=device,
        lr=lr
    )
    
    logger.info("training completed")
    
    logger.info("generating samples")
    samples = generate_samples(model, ddpm, num_samples=16, device=device)
    logger.info(f"samples generated: {samples.shape}")
