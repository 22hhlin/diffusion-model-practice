"""
Gaussian Diffusion (DDPM) for latent space.
Forward: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
Reverse: p(x_{t-1} | x_t) via learned noise prediction
"""
import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recip_m1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: sample x_t given x_0."""
        if noise is None:
            noise = torch.randn_like(x_0)
        return (self.sqrt_alphas_cumprod[t][:, None, None, None] * x_0 +
                self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise)

    def predict_x0_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise."""
        return (self.sqrt_recip_alphas_cumprod[t][:, None, None, None] * x_t -
                self.sqrt_recip_m1_alphas_cumprod[t][:, None, None, None] * noise)

    def q_posterior_mean(self, x_0, x_t, t):
        """Compute posterior mean q(x_{t-1} | x_t, x_0)."""
        return (self.posterior_mean_coef1[t][:, None, None, None] * x_0 +
                self.posterior_mean_coef2[t][:, None, None, None] * x_t)

    def p_mean_variance(self, x_t, t):
        """Compute predicted mean and variance for p(x_{t-1} | x_t)."""
        noise_pred = self.model(x_t, t)
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        mean = self.q_posterior_mean(x_0_pred, x_t, t)
        var = self.posterior_variance[t][:, None, None, None]
        log_var = self.posterior_log_variance_clipped[t][:, None, None, None]
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """Single reverse step: sample x_{t-1} from x_t."""
        mean, _, log_var = self.p_mean_variance(x_t, t)
        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, shape, device, temperature=1.0):
        """Full reverse process: sample from noise."""
        x = torch.randn(shape, device=device) * temperature
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        return x

    @torch.no_grad()
    def sample_with_progress(self, shape, device, every_n=100, temperature=1.0):
        """Sample and save intermediate results."""
        x = torch.randn(shape, device=device) * temperature
        intermediates = [x.cpu()]
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
            if t % every_n == 0 or t == 0:
                intermediates.append(x.cpu())
        return x, intermediates

    def training_loss(self, x_0):
        """Compute training loss (simple MSE on noise)."""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self.model(x_t, t)
        return nn.functional.mse_loss(noise_pred, noise)
