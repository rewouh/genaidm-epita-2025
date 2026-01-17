import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

"""
Paper: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

"We show that a certain parameterization of diffusion models reveals an equivalence 
with denoising score matching over multiple noise levels during training and with 
annealed Langevin dynamics during sampling."

This implementation follows the simplified training objective (Algorithm 1) and
sampling procedure (Algorithm 2) described in the paper.
"""


class DDPM:
    """
    Paper: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
    "A diffusion probabilistic model [...] is a parameterized Markov chain trained using
    variational inference to produce samples matching the data after finite time."
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear"
    ):
        # Paper Section 4: "We set T = 1000 for all experiments"
        self.num_timesteps = num_timesteps
        
        if schedule == "linear":
            # Paper Section 4: "We set the forward process variances to constants
            # increasing linearly from β₁ = 10⁻⁴ to βT = 0.02"
            self.betas = self._linear_schedule(beta_start, beta_end, num_timesteps)
        else:
            raise ValueError(f"schedule {schedule} not supported")
        
        # Paper Eq. (4): Define αₜ := 1 - βₜ and ᾱₜ := ∏ₛ₌₁ᵗ αₛ
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])
        
        # Precompute coefficients for the forward process q(xₜ|x₀)
        # Paper Eq. (4): q(xₜ|x₀) = N(xₜ; √ᾱₜ x₀, (1 - ᾱₜ)I)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Precompute coefficients for reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # Paper Eq. (7): β̃ₜ := (1 - ᾱₜ₋₁)/(1 - ᾱₜ) βₜ
        # Paper Section 3.2: "We set Σ_θ(x_t, t) = σ_t² I to untrained time dependent
        # constants. Experimentally, both σ_t² = β_t and σ_t² = β̃_t had similar results."
        # This implementation uses β̃_t (posterior_variance)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _linear_schedule(
        self,
        beta_start: float,
        beta_end: float,
        num_timesteps: int
    ) -> torch.Tensor:
        """
        Paper Section 4: "We set the forward process variances to constants
        increasing linearly from β₁ = 10⁻⁴ to βT = 0.02. These constants were
        chosen to be small relative to data scaled to [−1, 1], ensuring that
        reverse and forward processes have approximately the same functional form
        while keeping the signal-to-noise ratio at x_T as small as possible."
        """
        return torch.linspace(beta_start, beta_end, num_timesteps)
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paper Eq. (4): "A notable property of the forward process is that it admits
        sampling xₜ at an arbitrary timestep t in closed form"
        q(xₜ|x₀) = N(xₜ; √ᾱₜ x₀, (1 - ᾱₜ)I)
        
        This can be reparameterized as: xₜ(x₀, ε) = √ᾱₜ x₀ + √(1 - ᾱₜ) ε
        where ε ~ N(0, I) (from text below Eq. 9)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Paper Algorithm 2 (Sampling): "The complete sampling procedure"
        Paper Eq. (11): "We may choose the parameterization μ_θ(x_t, t) = 
        1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t))"
        
        This implements the sampling step:
        x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t)) + σ_t z
        where z ~ N(0, I) if t > 1, else z = 0
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        epsilon_theta = model(x_t, t_tensor)
        
        betas_t = self._extract(self.betas, t_tensor, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(
            self.sqrt_recip_alphas, t_tensor, x_t.shape
        )
        
        # Compute the mean using Eq. (11)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * epsilon_theta / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(
                self.posterior_variance, t_tensor, x_t.shape
            )
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: str = "cpu",
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Paper Algorithm 2 (Sampling):
        "1: x_T ~ N(0, I)
         2: for t = T, ..., 1 do
         3:   z ~ N(0, I) if t > 1, else z = 0
         4:   x_{t-1} = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t)) + σ_t z
         5: end for
         6: return x_0"
        
        Paper Section 3.2: "Algorithm 2 [...] resembles Langevin dynamics with
        ε_θ as a learned gradient of the data density."
        """
        model.eval()
        
        x_t = torch.randn(shape, device=device)
        
        if return_trajectory:
            trajectory = [x_t.cpu()]
        
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                x_t = self.p_sample(model, x_t, t, device)
                
                if return_trajectory:
                    trajectory.append(x_t.cpu())
        
        if return_trajectory:
            return torch.stack(trajectory)
        return x_t
    
    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def compute_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Paper Algorithm 1 (Training):
        "1: repeat
         2:   x_0 ~ q(x_0)
         3:   t ~ Uniform({1, ..., T})
         4:   ε ~ N(0, I)
         5:   Take gradient descent step on ∇_θ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²
         6: until converged"
        
        Paper Eq. (14): "L_simple(θ) := E_{t,x_0,ε}[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]"
        
        Paper Section 3.4: "We found it beneficial to sample quality (and simpler to 
        implement) to train on [...] simplified objective"
        """
        batch_size = x_0.shape[0]
        
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        noise = torch.randn_like(x_0)
        
        x_t, _ = self.q_sample(x_0, t, noise)
        
        epsilon_theta = model(x_t, t)
        
        loss = F.mse_loss(epsilon_theta, noise)
        
        return loss
    
    def forward_trajectory(
        self,
        x_0: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Paper Eq. (2): "The approximate posterior q(x_{1:T}|x_0), called the forward
        process or diffusion process, is fixed to a Markov chain that gradually adds
        Gaussian noise to the data according to a variance schedule β_1, ..., β_T"
        
        This demonstrates how "the diffusion consists of small amounts of Gaussian noise"
        and shows the "progressive lossy compression" mentioned in Section 4.3
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        timesteps = np.linspace(0, self.num_timesteps - 1, num_steps, dtype=int)
        trajectory = []
        
        noise = torch.randn_like(x_0)
        
        for t in timesteps:
            t_tensor = torch.tensor([t], dtype=torch.long)
            x_t, _ = self.q_sample(x_0, t_tensor, noise)
            trajectory.append(x_t)
        
        return torch.stack(trajectory)
