"""
This file provides visualization utilities for DDPM training and generation.

It includes functions to visualize:
- Forward trajectory: how images are progressively corrupted with noise
- Reverse trajectory: how noise is progressively removed to generate images
- Generated samples: grids of final generated images
- Training curves: loss evolution during training
- Noise prediction: comparison of true vs predicted noise
- Timestep comparisons: effect of different diffusion timestep counts

All visualizations can be saved as images or animated GIFs for analysis.
"""

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
    """
    Paper Eq. (2): "The approximate posterior q(x_{1:T}|x_0), called the forward
    process or diffusion process, is fixed to a Markov chain that gradually adds
    Gaussian noise to the data according to a variance schedule β_1, ..., β_T"
    """
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
    
    plt.close()


def visualize_reverse_trajectory(
    trajectory: torch.Tensor,
    save_path: Optional[str] = None,
    num_steps_to_show: int = 10,
    figsize: tuple = (15, 3)
):
    """
    Paper Section 2: "Transitions of this chain are learned to reverse a diffusion
    process, which is a Markov chain that gradually adds noise to the data in the
    opposite direction of sampling until signal is destroyed."
    
    Paper Section 3.2: "The complete sampling procedure, Algorithm 2, resembles
    Langevin dynamics with ε_θ as a learned gradient of the data density."
    
    Paper Section 4.3: "Large scale image features appear first and details appear last."
    """
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
    
    plt.close()


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
    
    plt.close()


def compare_timesteps(
    samples_list: List[torch.Tensor],
    timesteps_list: List[int],
    save_path: Optional[str] = None,
    num_samples_per_config: int = 8,
    figsize: tuple = (15, 10)
):
    """
    Paper Section 4.3: "We can therefore interpret the Gaussian diffusion model as
    a kind of autoregressive model with a generalized bit ordering that cannot be
    expressed by reordering data coordinates."
    
    Paper Section 4.3: "Gaussian diffusions can be made shorter for fast sampling
    or longer for model expressiveness."
    """
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
    
    plt.close()


def plot_training_curves(
    losses: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    iterations = np.arange(len(losses))
    
    ax1.plot(losses, alpha=0.3, color='#1f77b4', label='Perte brute', linewidth=0.5)
    
    window = min(100, len(losses) // 10)
    if window > 1:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(iterations[window-1:], moving_avg, 
                color='#ff7f0e', linewidth=2, label=f'Moyenne mobile ({window})')
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Itération')
    ax1.set_ylabel('MSE Loss (Log Scale)')
    ax1.set_title('Convergence Globale (Échelle Log)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    start_idx = int(len(losses) * 0.02) 
    if start_idx < len(losses):
        zoom_losses = losses[start_idx:]
        zoom_iter = iterations[start_idx:]
        
        ax2.plot(zoom_iter, zoom_losses, alpha=0.3, color='#1f77b4', linewidth=0.5)
        
        if window > 1 and len(moving_avg) > start_idx:
            ax2.plot(zoom_iter[window-1:], moving_avg[start_idx:], 
                    color='#ff7f0e', linewidth=2, label='Moyenne mobile')
            
        ax2.set_xlabel('Itération')
        ax2.set_ylabel('MSE Loss (Linear)')
        ax2.set_title(f'Apprentissage Fin (Zoom après itération {start_idx})')
        ax2.grid(True, alpha=0.3)
        
        if len(zoom_losses) > 0:
            y_max = np.percentile(zoom_losses, 95)
            ax2.set_ylim(0, y_max * 1.1)

    plt.suptitle('Dynamique d\'entraînement DDPM', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"training curves saved: {save_path}")
    
    plt.close()


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
    
    plt.close()


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
