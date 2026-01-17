import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
import os
import logging

logger = logging.getLogger(__name__)


def visualize_forward_trajectory(
    trajectory: torch.Tensor,
    save_path: Optional[str] = None,
    num_steps_to_show: int = 10,
    figsize: tuple = (15, 3)
):
    num_timesteps = trajectory.shape[0]
    indices = np.linspace(0, num_timesteps - 1, num_steps_to_show, dtype=int)
    
    fig, axes = plt.subplots(1, num_steps_to_show, figsize=figsize)
    
    for idx, ax in enumerate(axes):
        t_idx = indices[idx]
        img = trajectory[t_idx, 0, 0].cpu().numpy()
        
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f't = {t_idx}')
        ax.axis('off')
    
    plt.suptitle('forward trajectory: progressive noise addition', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"forward trajectory saved: {save_path}")
    
    plt.show()


def visualize_reverse_trajectory(
    trajectory: torch.Tensor,
    save_path: Optional[str] = None,
    num_steps_to_show: int = 10,
    figsize: tuple = (15, 3)
):
    num_timesteps = trajectory.shape[0]
    indices = np.linspace(0, num_timesteps - 1, num_steps_to_show, dtype=int)
    
    fig, axes = plt.subplots(1, num_steps_to_show, figsize=figsize)
    
    for idx, ax in enumerate(axes):
        t_idx = indices[idx]
        img = trajectory[t_idx, 0, 0].cpu().numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f't = {num_timesteps - t_idx - 1}')
        ax.axis('off')
    
    plt.suptitle('reverse trajectory: progressive denoising', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"reverse trajectory saved: {save_path}")
    
    plt.show()


def visualize_generated_samples(
    samples: torch.Tensor,
    save_path: Optional[str] = None,
    nrow: int = 8,
    figsize: tuple = (12, 12)
):
    num_samples = samples.shape[0]
    ncol = (num_samples + nrow - 1) // nrow
    
    fig, axes = plt.subplots(ncol, nrow, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(len(axes)):
        if idx < num_samples:
            img = samples[idx, 0].cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[idx].axis('off')
        else:
            axes[idx].axis('off')
    
    plt.suptitle(f'generated samples (n={num_samples})', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"samples saved: {save_path}")
    
    plt.show()


def compare_timesteps(
    samples_list: List[torch.Tensor],
    timesteps_list: List[int],
    save_path: Optional[str] = None,
    num_samples_per_config: int = 8,
    figsize: tuple = (15, 10)
):
    num_configs = len(samples_list)
    
    fig, axes = plt.subplots(num_configs, num_samples_per_config, figsize=figsize)
    
    if num_configs == 1:
        axes = axes.reshape(1, -1)
    
    for config_idx, (samples, timesteps) in enumerate(zip(samples_list, timesteps_list)):
        for sample_idx in range(num_samples_per_config):
            if sample_idx < samples.shape[0]:
                img = samples[sample_idx, 0].cpu().numpy()
                axes[config_idx, sample_idx].imshow(img, cmap='gray', vmin=-1, vmax=1)
            axes[config_idx, sample_idx].axis('off')
        
        axes[config_idx, 0].set_ylabel(f'T = {timesteps}', fontsize=12, rotation=0, 
                                        labelpad=40, va='center')
    
    plt.suptitle('comparison: effect of timestep count on quality', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"comparison saved: {save_path}")
    
    plt.show()


def plot_training_curves(
    losses: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5)
):
    plt.figure(figsize=figsize)
    plt.plot(losses, alpha=0.7, linewidth=0.5)
    
    window = min(100, len(losses) // 10)
    if window > 1:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), moving_avg, 
                linewidth=2, label=f'moving average (window={window})')
        plt.legend()
    
    plt.xlabel('iteration')
    plt.ylabel('mse loss')
    plt.title('ddpm training curve')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"training curves saved: {save_path}")
    
    plt.show()


def visualize_noise_prediction(
    x_t: torch.Tensor,
    true_noise: torch.Tensor,
    predicted_noise: torch.Tensor,
    t: int,
    save_path: Optional[str] = None,
    num_samples: int = 4,
    figsize: tuple = (12, 8)
):
    num_samples = min(num_samples, x_t.shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        axes[i, 0].imshow(x_t[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title('noisy image' if i == 0 else '')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_noise[i, 0].cpu().numpy(), cmap='gray', vmin=-3, vmax=3)
        axes[i, 1].set_title('true noise' if i == 0 else '')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(predicted_noise[i, 0].cpu().numpy(), cmap='gray', vmin=-3, vmax=3)
        axes[i, 2].set_title('predicted noise' if i == 0 else '')
        axes[i, 2].axis('off')
    
    plt.suptitle(f'noise prediction at t = {t}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"prediction visualization saved: {save_path}")
    
    plt.show()


def create_gif_from_trajectory(
    trajectory: torch.Tensor,
    save_path: str,
    fps: int = 30,
    sample_idx: int = 0
):
    try:
        from PIL import Image
        
        frames = []
        for t in range(trajectory.shape[0]):
            img = trajectory[t, sample_idx, 0].cpu().numpy()
            
            if img.min() < 0:
                img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            else:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            
            frames.append(Image.fromarray(img))
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,
            loop=0
        )
        logger.info(f"gif saved: {save_path}")
    except ImportError:
        logger.warning("pil not available, cannot create gif")
