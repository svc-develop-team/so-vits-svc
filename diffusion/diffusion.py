from collections import deque
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    def noise():
        return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def linear_beta_schedule(timesteps, max_beta=0.02):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


class GaussianDiffusion(nn.Module):
    def __init__(self, 
                denoise_fn, 
                out_dims=128,
                timesteps=1000, 
                k_step=1000,
                max_beta=0.02,
                spec_min=-12, 
                spec_max=2):
        
        super().__init__()
        self.denoise_fn = denoise_fn
        self.out_dims = out_dims
        betas = beta_schedule['linear'](timesteps, max_beta=max_beta)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.k_step = k_step if k_step>0 and k_step<timesteps else timesteps

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor([spec_min])[None, None, :out_dims])
        self.register_buffer('spec_max', torch.FloatTensor([spec_max])[None, None, :out_dims])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_ddim(self, x, t, interval, cond):
        """
        Use the DDIM method from
        """
        a_t = extract(self.alphas_cumprod, t, x.shape)
        a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)

        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = a_prev.sqrt() * (x / a_t.sqrt() + (((1 - a_prev) / a_prev).sqrt()-((1 - a_t) / a_t).sqrt()) * noise_pred)
        return x_prev

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                    a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, loss_type='l2'):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)

        if loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, 
                condition, 
                gt_spec=None, 
                infer=True, 
                infer_speedup=10, 
                method='dpm-solver',
                k_step=300,
                use_tqdm=True):
        """
            conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            spec = self.norm_spec(gt_spec)
            t = torch.randint(0, self.k_step, (b,), device=device).long()
            norm_spec = spec.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            return self.p_losses(norm_spec, t, cond=cond)
        else:
            shape = (cond.shape[0], 1, self.out_dims, cond.shape[2])
            
            if gt_spec is None:
                t = self.k_step
                x = torch.randn(shape, device=device)
            else:
                t = k_step
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]
                x = self.q_sample(x_start=norm_spec, t=torch.tensor([t - 1], device=device).long())
                        
            if method is not None and infer_speedup > 1:
                if method == 'dpm-solver' or method == 'dpm-solver++':
                    from .dpm_solver_pytorch import (
                        DPM_Solver,
                        NoiseScheduleVP,
                        model_wrapper,
                    )
                    # 1. Define the noise schedule.
                    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])

                    # 2. Convert your discrete-time `model` to the continuous-time
                    # noise prediction model. Here is an example for a diffusion model
                    # `model` with the noise prediction type ("noise") .
                    def my_wrapper(fn):
                        def wrapped(x, t, **kwargs):
                            ret = fn(x, t, **kwargs)
                            if use_tqdm:
                                self.bar.update(1)
                            return ret

                        return wrapped

                    model_fn = model_wrapper(
                        my_wrapper(self.denoise_fn),
                        noise_schedule,
                        model_type="noise",  # or "x_start" or "v" or "score"
                        model_kwargs={"cond": cond}
                    )

                    # 3. Define dpm-solver and sample by singlestep DPM-Solver.
                    # (We recommend singlestep DPM-Solver for unconditional sampling)
                    # You can adjust the `steps` to balance the computation
                    # costs and the sample quality.
                    if method == 'dpm-solver':
                        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    elif method == 'dpm-solver++':
                        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                        
                    steps = t // infer_speedup
                    if use_tqdm:
                        self.bar = tqdm(desc="sample time step", total=steps)
                    x = dpm_solver.sample(
                        x,
                        steps=steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                    if use_tqdm:
                        self.bar.close()
                elif method == 'pndm':
                    self.noise_list = deque(maxlen=4)
                    if use_tqdm:
                        for i in tqdm(
                                reversed(range(0, t, infer_speedup)), desc='sample time step',
                                total=t // infer_speedup,
                        ):
                            x = self.p_sample_plms(
                                x, torch.full((b,), i, device=device, dtype=torch.long),
                                infer_speedup, cond=cond
                            )
                    else:
                        for i in reversed(range(0, t, infer_speedup)):
                            x = self.p_sample_plms(
                                x, torch.full((b,), i, device=device, dtype=torch.long),
                                infer_speedup, cond=cond
                            )
                elif method == 'ddim':
                    if use_tqdm:
                        for i in tqdm(
                                reversed(range(0, t, infer_speedup)), desc='sample time step',
                                total=t // infer_speedup,
                        ):
                            x = self.p_sample_ddim(
                                x, torch.full((b,), i, device=device, dtype=torch.long),
                                infer_speedup, cond=cond
                            )
                    else:
                        for i in reversed(range(0, t, infer_speedup)):
                            x = self.p_sample_ddim(
                                x, torch.full((b,), i, device=device, dtype=torch.long),
                                infer_speedup, cond=cond
                            )
                elif method == 'unipc':
                    from .uni_pc import NoiseScheduleVP, UniPC, model_wrapper
                    # 1. Define the noise schedule.
                    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])

                    # 2. Convert your discrete-time `model` to the continuous-time
                    # noise prediction model. Here is an example for a diffusion model
                    # `model` with the noise prediction type ("noise") .
                    def my_wrapper(fn):
                        def wrapped(x, t, **kwargs):
                            ret = fn(x, t, **kwargs)
                            if use_tqdm:
                                self.bar.update(1)
                            return ret

                        return wrapped

                    model_fn = model_wrapper(
                        my_wrapper(self.denoise_fn),
                        noise_schedule,
                        model_type="noise",  # or "x_start" or "v" or "score"
                        model_kwargs={"cond": cond}
                    )

                    # 3. Define uni_pc and sample by multistep UniPC.
                    # You can adjust the `steps` to balance the computation
                    # costs and the sample quality.
                    uni_pc = UniPC(model_fn, noise_schedule, variant='bh2')

                    steps = t // infer_speedup
                    if use_tqdm:
                        self.bar = tqdm(desc="sample time step", total=steps)
                    x = uni_pc.sample(
                        x,
                        steps=steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                    )
                    if use_tqdm:
                        self.bar.close()
                else:
                    raise NotImplementedError(method)
            else:
                if use_tqdm:
                    for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                        x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
                else:
                    for i in reversed(range(0, t)):
                        x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            x = x.squeeze(1).transpose(1, 2)  # [B, T, M]
            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
