import torch
import os
import argparse
from pathlib import Path
from datetime import datetime
import logging

from genaidm.ddpm_mnist import DDPM
from genaidm.unet_model import SimpleUNet
from genaidm.train import train_ddpm, get_mnist_dataloader, generate_samples, load_checkpoint
from genaidm.visualization import (
    visualize_forward_trajectory,
    visualize_reverse_trajectory,
    visualize_generated_samples,
    compare_timesteps,
    plot_training_curves,
    visualize_noise_prediction,
    create_gif_from_trajectory
)
from genaidm.evaluation import evaluate_generated_samples, get_real_samples


def setup_logging(output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"ddpm_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"logging initialized, output directory: {output_dir}")
    logging.info(f"log file: {log_file}")
    return log_file


def experiment_forward_noising(
    ddpm: DDPM,
    dataloader,
    output_dir: str = "outputs/forward"
):
    logging.info("starting forward noising experiment")
    
    os.makedirs(output_dir, exist_ok=True)
    
    images, _ = next(iter(dataloader))
    x_0 = images[0:1] * 2 - 1
    
    logging.info("generating forward trajectory")
    trajectory = ddpm.forward_trajectory(x_0, num_steps=10)
    
    save_path = os.path.join(output_dir, "forward_trajectory.png")
    visualize_forward_trajectory(trajectory, save_path=save_path, num_steps_to_show=10)
    
    gif_path = os.path.join(output_dir, "forward_trajectory.gif")
    trajectory_full = ddpm.forward_trajectory(x_0, num_steps=100)
    create_gif_from_trajectory(trajectory_full, gif_path, fps=20)
    
    logging.info(f"results saved to {output_dir}")


def experiment_train_model(
    model: torch.nn.Module,
    ddpm: DDPM,
    dataloader,
    device: str,
    num_epochs: int = 20,
    output_dir: str = "outputs/training"
):
    logging.info("starting training experiment")
    
    os.makedirs(output_dir, exist_ok=True)
    
    losses = train_ddpm(
        model=model,
        ddpm=ddpm,
        dataloader=dataloader,
        num_epochs=num_epochs,
        device=device,
        lr=2e-4,
        save_dir=os.path.join(output_dir, "checkpoints"),
        save_every=5
    )
    
    save_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(losses, save_path=save_path)
    
    logging.info(f"training completed, results in {output_dir}")
    
    return losses


def experiment_reverse_denoising(
    model: torch.nn.Module,
    ddpm: DDPM,
    device: str,
    output_dir: str = "outputs/reverse"
):
    logging.info("experiment 3: reverse process denoising")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("generating multiple reverse trajectories")
    num_trajectories = 4
    
    for i in range(num_trajectories):
        logging.info(f"trajectory {i+1}/{num_trajectories}")
        trajectory = generate_samples(
            model=model,
            ddpm=ddpm,
            num_samples=1,
            device=device,
            return_trajectory=True
        )
        
        save_path = os.path.join(output_dir, f"reverse_trajectory_{i+1}.png")
        visualize_reverse_trajectory(trajectory, save_path=save_path, num_steps_to_show=10)
        
        if i == 0:
            gif_path = os.path.join(output_dir, "reverse_trajectory.gif")
            create_gif_from_trajectory(trajectory, gif_path, fps=30)
    
    logging.info("generating trajectory grid")
    trajectories_grid = generate_samples(
        model=model,
        ddpm=ddpm,
        num_samples=8,
        device=device,
        return_trajectory=True
    )
    
    import matplotlib.pyplot as plt
    key_steps = [0, ddpm.num_timesteps // 4, ddpm.num_timesteps // 2, 
                 3 * ddpm.num_timesteps // 4, ddpm.num_timesteps - 1]
    
    fig, axes = plt.subplots(len(key_steps), 8, figsize=(16, 10))
    for step_idx, t in enumerate(key_steps):
        for sample_idx in range(8):
            img = trajectories_grid[t, sample_idx, 0].cpu().numpy()
            axes[step_idx, sample_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[step_idx, sample_idx].axis('off')
        axes[step_idx, 0].set_ylabel(f't={ddpm.num_timesteps - t - 1}', 
                                     fontsize=10, rotation=0, labelpad=30, va='center')
    
    plt.suptitle('progressive denoising (8 samples)', fontsize=14)
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "denoising_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"grid saved: {grid_path}")
    
    logging.info(f"results saved in {output_dir}")


def experiment_generate_samples(
    model: torch.nn.Module,
    ddpm: DDPM,
    device: str,
    num_samples: int = 64,
    output_dir: str = "outputs/samples"
):
    logging.info("experiment 4: sample generation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"generating {num_samples} samples...")
    samples = generate_samples(
        model=model,
        ddpm=ddpm,
        num_samples=num_samples,
        device=device
    )
    
    save_path = os.path.join(output_dir, f"generated_samples_{num_samples}.png")
    visualize_generated_samples(samples, save_path=save_path, nrow=8)
    
    logging.info("generating detailed trajectories...")
    detailed_trajectories = generate_samples(
        model=model,
        ddpm=ddpm,
        num_samples=4,
        device=device,
        return_trajectory=True
    )
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 10, figsize=(20, 8))
    timesteps_to_show = torch.linspace(0, ddpm.num_timesteps - 1, 10).long()
    
    for sample_idx in range(4):
        for step_idx, t in enumerate(timesteps_to_show):
            img = detailed_trajectories[t, sample_idx, 0].cpu().numpy()
            axes[sample_idx, step_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[sample_idx, step_idx].axis('off')
            if sample_idx == 0:
                axes[sample_idx, step_idx].set_title(f't={ddpm.num_timesteps - t - 1}', fontsize=8)
        axes[sample_idx, 0].set_ylabel(f'sample {sample_idx+1}', rotation=0, labelpad=40, va='center')
    
    plt.suptitle('complete denoising trajectories', fontsize=14)
    plt.tight_layout()
    detailed_path = os.path.join(output_dir, "detailed_trajectories.png")
    plt.savefig(detailed_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"detailed trajectories: {detailed_path}")
    
    logging.info("diversity analysis")
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(4):
        batch = generate_samples(model, ddpm, num_samples=8, device=device)
        for j in range(8):
            axes[i, j].imshow(batch[j, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(f'batch {i+1}', rotation=0, labelpad=30, va='center')
    
    plt.suptitle('diversity analysis (4 independent batches)', fontsize=14)
    plt.tight_layout()
    diversity_path = os.path.join(output_dir, "diversity_analysis.png")
    plt.savefig(diversity_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"diversity analysis: {diversity_path}")
    
    logging.info("creating animated gifs")
    for i in range(min(2, detailed_trajectories.shape[1])):
        gif_path = os.path.join(output_dir, f"denoising_sample_{i+1}.gif")
        create_gif_from_trajectory(detailed_trajectories[:, i:i+1], gif_path, fps=30, sample_idx=0)
        logging.info(f"gif {i+1}: {gif_path}")
    
    logging.info(f"samples saved in {output_dir}")
    
    return samples


def experiment_compare_timesteps(
    model_path: str,
    device: str,
    timesteps_list: list = [100, 250, 500, 1000],
    num_samples: int = 8,
    output_dir: str = "outputs/comparison"
):
    logging.info("experiment 5: timestep comparison")
    
    os.makedirs(output_dir, exist_ok=True)
    
    samples_list = []
    
    for T in timesteps_list:
        logging.info(f"generating with t={T} timesteps")
        
        ddpm_temp = DDPM(num_timesteps=T, beta_start=1e-4, beta_end=0.02)
        
        model = SimpleUNet(image_channels=1, base_channels=64)
        load_checkpoint(model_path, model, device)
        
        samples = generate_samples(
            model=model,
            ddpm=ddpm_temp,
            num_samples=num_samples,
            device=device
        )
        
        samples_list.append(samples)
        
        individual_path = os.path.join(output_dir, f"samples_T{T}.png")
        visualize_generated_samples(samples, save_path=individual_path, nrow=4)
    
    save_path = os.path.join(output_dir, "timesteps_comparison.png")
    compare_timesteps(
        samples_list=samples_list,
        timesteps_list=timesteps_list,
        save_path=save_path,
        num_samples_per_config=num_samples
    )
    
    logging.info("comparing denoising trajectories")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(len(timesteps_list), 10, figsize=(20, len(timesteps_list) * 2))
    
    for config_idx, T in enumerate(timesteps_list):
        logging.info(f"trajectory for t={T}")
        ddpm_temp = DDPM(num_timesteps=T, beta_start=1e-4, beta_end=0.02)
        model = SimpleUNet(image_channels=1, base_channels=64)
        load_checkpoint(model_path, model, device)
        
        trajectory = generate_samples(
            model=model,
            ddpm=ddpm_temp,
            num_samples=1,
            device=device,
            return_trajectory=True
        )
        
        indices = torch.linspace(0, T, 10).long()
        
        for step_idx, t_idx in enumerate(indices):
            img = trajectory[t_idx, 0, 0].cpu().numpy()
            ax = axes[config_idx, step_idx] if len(timesteps_list) > 1 else axes[step_idx]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if config_idx == 0:
                ax.set_title(f'{step_idx+1}/10', fontsize=8)
        
        ax = axes[config_idx, 0] if len(timesteps_list) > 1 else axes[0]
        ax.set_ylabel(f'T={T}', rotation=0, labelpad=30, va='center', fontsize=10)
    
    plt.suptitle('trajectory comparison for different timestep counts', fontsize=14)
    plt.tight_layout()
    traj_comp_path = os.path.join(output_dir, "trajectory_comparison.png")
    plt.savefig(traj_comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"trajectory comparison: {traj_comp_path}")
    
    logging.info(f"comparison saved in {output_dir}")


def experiment_evaluation(
    model: torch.nn.Module,
    ddpm: DDPM,
    device: str,
    num_samples: int = 1000,
    output_dir: str = "outputs/evaluation"
):
    logging.info("experiment 6: evaluation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"generating {num_samples} samples for evaluation")
    generated = generate_samples(
        model=model,
        ddpm=ddpm,
        num_samples=num_samples,
        device=device
    )
    
    logging.info("loading real samples")
    real_samples = get_real_samples(num_samples=num_samples)
    
    results = evaluate_generated_samples(
        generated_samples=generated,
        real_samples=real_samples,
        device=device
    )
    
    results_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("ddpm evaluation results\n")
        f.write("="*50 + "\n\n")
        f.write(f"number of samples: {num_samples}\n\n")
        f.write(f"average confidence: {results['avg_confidence']:.4f}\n\n")
        f.write("class distribution:\n")
        for i, prob in enumerate(results['class_distribution']):
            f.write(f"  class {i}: {prob*100:.2f}%\n")
        if 'fid' in results:
            f.write(f"\nfid score: {results['fid']:.2f}\n")
    
    logging.info(f"results saved in {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="ddpm on mnist - complete experiments")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "forward", "train", "reverse", "generate", 
                               "compare", "evaluate"],
                       help="experiment mode")
    parser.add_argument("--epochs", type=int, default=20,
                       help="number of training epochs")
    parser.add_argument("--timesteps", type=int, default=1000,
                       help="number of diffusion timesteps")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="batch size")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="path to checkpoint to load")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="output directory")
    parser.add_argument("--device", type=str, default=None,
                       help="device (cpu or cuda)")
    
    args = parser.parse_args()
    
    setup_logging(args.output_dir)
    
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logging.info("ddpm on mnist - experiments")
    logging.info(f"device: {device}")
    logging.info(f"timesteps: {args.timesteps}")
    logging.info(f"mode: {args.mode}")
    
    ddpm = DDPM(num_timesteps=args.timesteps, beta_start=1e-4, beta_end=0.02, schedule="linear")
    model = SimpleUNet(image_channels=1, base_channels=64)
    
    dataloader = get_mnist_dataloader(batch_size=args.batch_size)
    
    checkpoint_path = None
    
    if args.mode in ["all", "forward"]:
        experiment_forward_noising(ddpm, dataloader, 
                                  output_dir=os.path.join(args.output_dir, "forward"))
    
    if args.mode in ["all", "train"]:
        losses = experiment_train_model(model, ddpm, dataloader, device, 
                                       num_epochs=args.epochs,
                                       output_dir=os.path.join(args.output_dir, "training"))
        checkpoint_path = Path(args.output_dir) / "training" / "checkpoints" / "ddpm_final.pt"
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.mode == "train" and checkpoint_path is None:
        checkpoint_path = Path(args.output_dir) / "training" / "checkpoints" / "ddpm_final.pt"
    
    if checkpoint_path and checkpoint_path.exists():
        load_checkpoint(str(checkpoint_path), model, device)
    elif args.mode not in ["forward", "train"]:
        logging.warning("no checkpoint found, model is not trained")
        logging.info(f"searched path: {checkpoint_path}")
        logging.info("use --checkpoint <path> or train first with --mode train")
        return
    
    if args.mode in ["all", "reverse"]:
        experiment_reverse_denoising(model, ddpm, device,
                                    output_dir=os.path.join(args.output_dir, "reverse"))
    
    if args.mode in ["all", "generate"]:
        samples = experiment_generate_samples(model, ddpm, device, num_samples=64,
                                             output_dir=os.path.join(args.output_dir, "samples"))
    
    if args.mode in ["all", "compare"]:
        if checkpoint_path and checkpoint_path.exists():
            experiment_compare_timesteps(str(checkpoint_path), device, 
                                        timesteps_list=[100, 250, 500, 1000],
                                        output_dir=os.path.join(args.output_dir, "comparison"))
    
    if args.mode in ["all", "evaluate"]:
        results = experiment_evaluation(model, ddpm, device, num_samples=1000,
                                       output_dir=os.path.join(args.output_dir, "evaluation"))
    
    logging.info("all experiments completed")
    logging.info(f"results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
