# DDPM on MNIST

**Authors:** Arthur Hamard, Arthur Guelennoc, Léo Sambrook, Etienne Senigout, Pierre Braud, Paul Pazart

## Description

Implementation of Denoising Diffusion Probabilistic Models (DDPM) for generating MNIST digits. This project includes the complete pipeline: forward diffusion process, model training, sample generation, and evaluation metrics (FID, classifier confidence).

## Repository Structure

```
src/
├── genaidm/           # Core DDPM implementation
│   ├── ddpm_mnist.py  # DDPM algorithm (forward/reverse process)
│   ├── unet_model.py  # U-Net architecture for denoising
│   ├── train.py       # Training pipeline
│   ├── evaluation.py  # Evaluation metrics (FID, classifier)
│   └── visualization.py # Visualization utilities
├── main.py            # Main entry point for experiments
├── notebook.ipynb     # Interactive notebook
└── outputs/           # Generated results (samples, checkpoints, etc.)
```

## Installation

```bash
uv sync
```

## Commands

### Training
```bash
uv run python src/main.py --mode train --epochs 100
```

### Generate samples
```bash
uv run python src/main.py --mode generate --checkpoint outputs/training/checkpoints/ddpm_final.pt
```

### Run all experiments
```bash
uv run python src/main.py --mode all --epochs 100
```

### Available modes
- `forward`: Visualize forward diffusion process
- `train`: Train the model
- `reverse`: Visualize reverse denoising process
- `generate`: Generate new samples
- `compare`: Compare different timestep counts
- `evaluate`: Compute evaluation metrics
- `all`: Run all experiments

### Options
- `--epochs`: Number of training epochs (default: 100)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--batch-size`: Batch size (default: 128)
- `--checkpoint`: Path to model checkpoint
- `--output-dir`: Output directory (default: outputs)
- `--device`: Device to use (cpu/cuda, auto-detected by default)

