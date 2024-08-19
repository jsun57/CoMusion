import numpy as np
import math
from functools import partial
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def to_torch_tensor(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result)
        return result
    return wrapper

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

@to_torch_tensor
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "cosine":           # 0.008
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / (1 + 0.008) * math.pi / 2) ** 2,
        )
    elif schedule_name == "ours":           # NOTE: ours: 1: poposed variance scheduler
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 1) / (1 + 1) * np.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / max(0.00001, alpha_bar(t1)), max_beta)) # the max is to prevent singularities
    return np.array(betas)

def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# gaussian diffusion class

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        cfg,
        future_motion_size, # [T_pred, N*C=num_nodes]
        timesteps=50,
        loss_type='l1',
        objective='pred_x0',
        beta_schedule='ours',
        history_weight=1,
        future_weight=20,
        st_loss_weight=None,   # [T_pred, N*C=num_nodes] if exists
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.cfg = cfg

        self.future_motion_size = future_motion_size

        self.objective = objective
        self.st_loss_weight = st_loss_weight

        # loss component related
        self.his_weight = history_weight
        self.pre_weight = future_weight
        
        self.var_large = True
        assert objective == 'pred_x0'   # NOTE: ours direct prediction target

        betas = get_named_beta_schedule(beta_schedule, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = timesteps # default num sampling timesteps to number of timesteps at training

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_0_history, act=None, clip_x_start=False, uncond=False):
        """
        x: [b, t_pred, nc], t: [b, 1], x_0_history: [b, t_his, nc], act: [b, 1]
        """        
        model_output = self.model(x, t, x_0_history, act, uncond)                   # [b, t_all, NC] / [b, s, t_all, NC]
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        t_pred = self.future_motion_size[0]

        x_start = model_output[:,-t_pred:,:] if model_output.ndim == 3 else model_output[:,0,-t_pred:,:]
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_0_history, act=None, clip_denoised=False, uncond=False):
        preds = self.model_predictions(x, t, x_0_history, act, uncond=uncond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        if self.var_large:
            model_variance = extract(torch.cat((self.posterior_variance[1].unsqueeze(0), self.betas[1:])), t, x.shape)
            model_log_variance = extract(torch.log(torch.cat((self.posterior_variance[1].unsqueeze(0), self.betas[1:]))), t, x.shape)

        model_mean, _, _ = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, model_variance, model_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_0_history, act=None, clip_denoised=False, uncond=False):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_0_history=x_0_history, \
            act=act, clip_denoised=clip_denoised, uncond=uncond)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_future = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_future, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_0_history, shape, act=None, return_all_timesteps=False, return_x_starts=False, clip_denoised=False, uncond=False):
        batch, device = shape[0], self.betas.device

        motion_future = torch.randn(shape, device = device)
        motion_futures = [motion_future]
        x_starts = []

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            motion_future, x_start = self.p_sample(motion_future, t, x_0_history, act, clip_denoised, uncond)
            motion_futures.append(motion_future)
            x_starts.append(x_start)

        ret = motion_future if not return_all_timesteps else torch.stack(motion_futures, dim=1)
        x_starts = torch.stack(x_starts, dim=1)

        if not return_x_starts:
            return ret 
        else:
            return ret, x_starts    # [sample_num, diff_steps, t_pred, nc]

    @torch.no_grad()
    def sample(self, x_0_history, act=None, batch_size=16, return_all_timesteps=False, return_x_starts=False, clip_denoised=False, uncond=False):
        future_motion_length, num_features = self.future_motion_size
        sample_fn = self.p_sample_loop
        return sample_fn(x_0_history, (batch_size, future_motion_length, num_features), act, \
            return_all_timesteps=return_all_timesteps, return_x_starts=return_x_starts, clip_denoised=clip_denoised, uncond=uncond)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def smooth_loss_target(self, target):
        """target: [b, div_k, t_all, NC] -> x_shape: [b, div_k, s, t_all, NC]"""
        def smooth(data, seq_in, seq_out):
            """ smoothing data on t axis: [b, div_k, t_all, NC]"""
            smooth_data = data.clone()
            for i in range(seq_in, seq_in+seq_out):
                smooth_data[...,i,:] = torch.mean(data[...,seq_in:i+1,:], dim=-2)
            return smooth_data

        def recursive_smooth(data, seq_in, seq_out, num_iter=1):
            """data: orig: [b, div_k, t_all, NC]"""
            ret = []
            for idx in range(num_iter):
                if idx == 0:
                    ret.append(data.clone())
                else:
                    smoothed = smooth(ret[-1], seq_in, seq_out)
                    ret.append(smoothed.clone())
            ret = torch.stack(ret).permute(1, 2, 0, 3, 4)
            return ret

        all_smoothed_gt = recursive_smooth(target, self.model.input_n, self.model.output_n, self.model.outer_stage) # [b, div_k, s, t_all, nc]
        return all_smoothed_gt
            
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            def loss_function(model_out, target, weights=None, reduction='none'):
                """[b, k, t_all, nc], [b, k, t_all, nc]"""
                if model_out.ndim == 5:
                    target = self.smooth_loss_target(target)
                    loss = F.l1_loss(model_out, target, reduction='none')
                    loss = reduce(loss, 'b k s t d -> b k t d', 'mean')
                else:
                    loss = F.l1_loss(model_out, target, reduction='none')   
                if weights is not None:
                    weights = weights[None,None,:,:].to(model_out.device)    # [1, 1, t_all, nc]
                    loss = loss * weights
                return loss  
            return loss_function
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def process_div_dim(self, *args, b, merge=True):
        if merge:
            return [rearrange(x, 'b v ... -> (v b) ...') for x in args]
        else:
            return [rearrange(x, '(v b) ... -> b v ...', b=b) for x in args]

    def p_losses(self, x_start, t, x_0_history, act=None, noise=None, div_k=1, uncond=False, mmgt=None):
        """
        x_start: [b, t_pred, nc]; t: [b, 1]; x_0_history: [b, t_his, nc]
        act: [b, 1]
        """
        b, l, d = x_start.shape
        x_start = x_start.unsqueeze(1).repeat(1, div_k, 1, 1)               # [b, div_k, t_pred, nc]
        noise = default(noise, lambda: torch.randn_like(x_start))           # [b, div_k, t_pred, nc]
        raw_x_0_history = x_0_history.unsqueeze(1).repeat(1, div_k, 1, 1)   # [b, div_k, t_his, nc]
        x_0_history = raw_x_0_history                                       # [b, div_k, t_his, nc]

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)            # [b, div_k, t_pred, nc]

        # No self-conditioning
        x, x_0_history = self.process_div_dim(x, x_0_history, b=b, merge=True)  # 3 dims

        # predict and take gradient step
        x_0_history = self.process_div_dim(x_0_history, b=b, merge=False)[0]
        x_0_history = self.process_div_dim(x_0_history, b=b, merge=True)[0]
        
        model_out = self.model(x, t.repeat(div_k), x_0_history, act.repeat(div_k, 1), uncond=uncond) if exists(act) \
        else self.model(x, t.repeat(div_k), x_0_history, None, uncond=uncond)
        model_out = self.process_div_dim(model_out, b=b, merge=False)[0]    # [b, div_k, t_all, nc] / [b, div_k, s, t_all, nc]

        target = torch.cat((raw_x_0_history, x_start), dim=-2)              # [b, div_k, t_all, nc]

        t_his = x_0_history.shape[1]
        t_pred, _ = self.future_motion_size

        loss = self.loss_fn(model_out, target, weights=self.st_loss_weight, reduction='none')   # [b, div_k, t_all, nc] / [b, div_k, t_all, n]

        # 1. history recovery: position based
        his_loss = loss[...,:t_his,:]
        his_loss = reduce(his_loss, 'b ... -> b (...)', 'mean')
        his_loss = his_loss.mean()

        # 2. future construction: position based: incorprate min in div_k for implicit diversity
        pre_loss = loss[...,-t_pred:,:]
        pre_loss = reduce(pre_loss, 'b k t d -> b t d', 'min')
        pre_loss = reduce(pre_loss, 'b ... -> b (...)', 'mean')
        pre_loss = pre_loss.mean()
            
        total_loss = self.his_weight * his_loss + self.pre_weight * pre_loss
        
        extra = {
            'his_loss': self.his_weight * his_loss.item(),
            'pre_loss': self.pre_weight * pre_loss.item()
        }

        return total_loss, extra

    def forward(self, complete_motion, act=None, *args, **kwargs):
        """
        if exist, act: [b, 1] int labels
        """
        b, t_total, k = complete_motion.shape
        device, t_pred, _ = complete_motion.device, *self.future_motion_size

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()    # [b, 1] random ts in range [0, num_timesteps]

        x_start = complete_motion[:,-t_pred:,:] # [b, t_pred, nc]
        x_0_history = complete_motion[:,:-t_pred,:]

        return self.p_losses(x_start, t, x_0_history, act, *args, **kwargs)
