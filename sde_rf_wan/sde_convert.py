"""
sde_convert.py — Inference-Time SDE Conversion for Flow Models

Implements the core contribution of the paper:
  - Eq.7: Reverse SDE with freely chosen diffusion coefficient
  - Eq.8: Score function derived from pretrained velocity field
  - Eq.9: Discrete-time proposal distribution (Euler-Maruyama)

All functions operate on the LINEAR interpolant: α_t = 1-t, σ_t = t.
"""

import torch
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================
# Interpolant definition (Sec. 4.1, Eq.5)
# ============================================================
# Linear: α_t = 1 - t,  σ_t = t
#          α̇_t = -1,     σ̇_t = 1

@dataclass
class LinearInterpolant:
    """Linear interpolant used by RF models (FLUX, SD3, Wan)."""

    @staticmethod
    def alpha(t: float) -> float:
        return 1.0 - t

    @staticmethod
    def sigma(t: float) -> float:
        return t

    @staticmethod
    def d_alpha(t: float) -> float:
        """Time derivative of α_t."""
        return -1.0

    @staticmethod
    def d_sigma(t: float) -> float:
        """Time derivative of σ_t."""
        return 1.0


# ============================================================
# Eq.8: Score function from velocity field
# ============================================================

def velocity_to_score(
    u_t: torch.Tensor,
    x_t: torch.Tensor,
    t: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute ∇log p_t(x_t) from the pretrained velocity field u_t.

    Paper Eq.8 (general form):

                  1     α_t · u_t(x_t)  -  α̇_t · x_t
        score = ─── · ────────────────────────────────
                 σ_t    α̇_t · σ_t  -  α_t · σ̇_t

    For linear interpolant (α_t=1-t, σ_t=t, α̇=-1, σ̇=1):

        numerator   = (1-t) · u_t  -  (-1) · x_t  =  (1-t)·u_t + x_t
        denominator = (-1)·t - (1-t)·1             =  -t - 1 + t  =  -1
        score       = (1/t) · [(1-t)·u_t + x_t] / (-1)
                    = -[(1-t)·u_t + x_t] / t
                    = [x_t + (1-t)·u_t] / (-t)

    Equivalently (and more intuitively via ε = x_t + (1-t)·v, x_0 = x_t - t·v):

        score = -(x_t - (1-t)·u_t) / t      ... wrong sign?

    Let me re-derive carefully:
        α_t = 1-t,  σ_t = t,  α̇_t = -1,  σ̇_t = 1

        num = α_t · u_t - α̇_t · x_t = (1-t)·u_t + x_t
        den = α̇_t · σ_t - α_t · σ̇_t = -t - (1-t) = -1

        score = (1/σ_t) · (num / den) = (1/t) · [(1-t)·u_t + x_t] / (-1)
              = -[(1-t)·u_t + x_t] / t

    Note: x_t = (1-t)·x_0 + t·ε, u_t = ε - x_0
        (1-t)·u_t + x_t = (1-t)(ε-x_0) + (1-t)x_0 + tε = (1-t)ε + tε = ε
        So score = -ε / t = -x_1 / σ_t

    This matches the known relationship: score = -ε / σ_t for DDPM-style models.

    Args:
        u_t: (B, C, ...) velocity prediction from pretrained model
        x_t: (B, C, ...) current noisy sample
        t: current timestep in [0, 1]
        eps: numerical stability for small t

    Returns:
        score: (B, C, ...) ∇log p_t(x_t)
    """
    interp = LinearInterpolant()
    alpha_t = interp.alpha(t)
    sigma_t = max(interp.sigma(t), eps)
    d_alpha_t = interp.d_alpha(t)
    d_sigma_t = interp.d_sigma(t)

    numerator = alpha_t * u_t - d_alpha_t * x_t
    denominator = d_alpha_t * sigma_t - alpha_t * d_sigma_t

    score = (1.0 / sigma_t) * (numerator / (denominator + eps))
    return score


# ============================================================
# Diffusion coefficient g_t (Appendix F, Table 8b)
# ============================================================

def diffusion_coeff(t: float, scale: float = 3.0) -> float:
    """Diffusion coefficient g_t = scale · t².

    Paper choice: g_t = 3·t² (Appendix F, Table 8b).

    Properties:
      - g(0) = 0: no noise at t=0 (clean image), smooth transition to ODE
      - g(1) = scale: maximum noise at t=1 (pure noise region)
      - Quadratic growth: gentle near clean, aggressive near noise
      - When scale=0: reduces to deterministic ODE sampling (Eq.6)

    Args:
        t: timestep in [0, 1]
        scale: multiplicative factor (paper uses 3.0)

    Returns:
        g_t: scalar diffusion coefficient
    """
    return scale * (t ** 2)


# ============================================================
# Eq.7: SDE drift coefficient
# ============================================================

def sde_drift(
    u_t: torch.Tensor,
    score: torch.Tensor,
    g_t: float,
) -> torch.Tensor:
    """Compute SDE drift f_t(x_t) from Eq.7.

        f_t(x_t) = u_t(x_t) - (g_t² / 2) · ∇log p_t(x_t)

    When g_t = 0, this reduces to f_t = u_t (deterministic ODE).

    Args:
        u_t: (B, C, ...) velocity from pretrained model
        score: (B, C, ...) score function (from velocity_to_score)
        g_t: diffusion coefficient at current timestep

    Returns:
        f_t: (B, C, ...) SDE drift
    """
    return u_t - 0.5 * (g_t ** 2) * score


# ============================================================
# Eq.9: One-step SDE sampling (Euler-Maruyama)
# ============================================================

@dataclass
class SDEStepOutput:
    """Output of a single SDE step."""
    x_next: torch.Tensor        # x_{t-Δt}: denoised sample
    drift: torch.Tensor         # f_t · Δt: deterministic component
    noise_coeff: float          # g_t · √Δt: noise scaling factor
    noise: torch.Tensor         # ε: the sampled (or codebook) noise
    score: torch.Tensor         # ∇log p_t: for diagnostics
    x0_pred: torch.Tensor       # posterior mean estimate


def sde_euler_maruyama_step(
    u_t: torch.Tensor,
    x_t: torch.Tensor,
    t_curr: float,
    t_next: float,
    noise: Optional[torch.Tensor] = None,
    g_scale: float = 3.0,
    score_eps: float = 1e-6,
) -> SDEStepOutput:
    """Single Euler-Maruyama step for the reverse SDE (Eq.9).

    Discrete form of Eq.7:

        x_{t-Δt} = x_t  -  f_t · Δt  +  g_t · √Δt · ε

    where:
        f_t = u_t - (g_t²/2) · ∇log p_t   (Eq.7)
        ε ~ N(0, I) or from codebook        ← THIS IS WHAT WE REPLACE

    The proposal distribution is (Eq.9):
        p_θ(x_{t-Δt} | x_t) = N(x_t - f_t·Δt,  g_t²·Δt · I)

    Args:
        u_t: velocity prediction from model at (x_t, t_curr)
        x_t: current sample
        t_curr: current timestep (larger, noisier)
        t_next: target timestep (smaller, cleaner)
        noise: if provided, use this instead of sampling N(0,I).
               This is the codebook entry in DDCM mode.
        g_scale: diffusion coefficient scaling factor
        score_eps: numerical stability

    Returns:
        SDEStepOutput with all components for analysis/debugging
    """
    delta_t = t_curr - t_next  # positive (going from noisy to clean)
    assert delta_t > 0, f"t_curr ({t_curr}) must be > t_next ({t_next})"

    # Eq.8: score from velocity
    score = velocity_to_score(u_t, x_t, t_curr, eps=score_eps)

    # Diffusion coefficient
    g_t = diffusion_coeff(t_curr, scale=g_scale)

    # Eq.7: SDE drift
    f_t = sde_drift(u_t, score, g_t)

    # Noise coefficient
    noise_coeff = g_t * (delta_t ** 0.5)

    # Sample or use provided noise (codebook entry)
    if noise is None:
        noise = torch.randn_like(x_t)

    # Eq.9: Euler-Maruyama step
    #   x_{t-Δt} = x_t - f_t · Δt + g_t · √Δt · ε
    x_next = x_t - f_t * delta_t + noise_coeff * noise

    # Posterior mean (value function approximation, Sec. 3.2)
    # x_{0|t} = x_t - t · u_t  (from linear interpolant)
    x0_pred = x_t - t_curr * u_t

    return SDEStepOutput(
        x_next=x_next,
        drift=f_t * delta_t,
        noise_coeff=noise_coeff,
        noise=noise,
        score=score,
        x0_pred=x0_pred,
    )


# ============================================================
# Full SDE sampling loop
# ============================================================

def sde_sample_loop(
    model_fn,
    x_init: torch.Tensor,
    timesteps: torch.Tensor,
    noise_list: Optional[list] = None,
    g_scale: float = 3.0,
) -> Tuple[torch.Tensor, list]:
    """Full reverse SDE sampling loop.

    Args:
        model_fn: callable(x_t, t) → u_t (velocity prediction)
        x_init: initial noise x_T ~ N(0, I)
        timesteps: descending sequence [t_0=1, t_1, ..., t_N=0]
        noise_list: if provided, use these noises instead of random.
                    Length must be len(timesteps)-1 (one per step).
                    Last step (to t=0) uses no noise.
        g_scale: diffusion coefficient scale

    Returns:
        x_final: generated sample
        all_noises: list of noise tensors used (for encoding)
    """
    x_t = x_init
    all_noises = []
    num_steps = len(timesteps) - 1

    for i in range(num_steps):
        t_curr = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        # Get velocity prediction
        u_t = model_fn(x_t, t_curr)

        # Last step: no noise (t_next ≈ 0)
        if t_next < 1e-6:
            # Pure ODE step to reach t=0
            delta_t = t_curr - t_next
            x_t = x_t - u_t * delta_t  # equivalent to x_t + u_t * (t_next - t_curr)
            all_noises.append(torch.zeros_like(x_t))
            break

        # Select noise
        noise = noise_list[i] if noise_list is not None else None

        # SDE step (Eq.9)
        output = sde_euler_maruyama_step(
            u_t=u_t,
            x_t=x_t,
            t_curr=t_curr,
            t_next=t_next,
            noise=noise,
            g_scale=g_scale,
        )

        x_t = output.x_next
        all_noises.append(output.noise)

    return x_t, all_noises


# ============================================================
# ODE baseline (Eq.6) for comparison
# ============================================================

def ode_sample_loop(
    model_fn,
    x_init: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Standard deterministic ODE sampling (Eq.6): dx = u_t dt.

    Used as baseline to verify SDE conversion preserves quality.
    """
    x_t = x_init

    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i].item()
        t_next = timesteps[i + 1].item()
        dt = t_next - t_curr  # negative

        u_t = model_fn(x_t, t_curr)
        x_t = x_t + u_t * dt  # Euler step

    return x_t


# ============================================================
# Timestep scheduling
# ============================================================

def linear_timesteps(num_steps: int, device: torch.device = None) -> torch.Tensor:
    """Uniform timesteps from 1→0."""
    return torch.linspace(1.0, 0.0, num_steps + 1, device=device)


def shifted_timesteps(
    num_steps: int,
    shift: float = 3.0,
    device: torch.device = None,
) -> torch.Tensor:
    """SD3-style shifted schedule (more steps in high-noise region).

    t_shifted = shift · t / (1 + (shift-1) · t)

    SD3 default shift=3.0. Wan uses shift=5.0 (720P) or 3.0 (480P).
    """
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    t_shifted = shift * t / (1.0 + (shift - 1.0) * t)
    return t_shifted
