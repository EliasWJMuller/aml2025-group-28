import torch
import torch.nn.functional as F
from tqdm import tqdm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def generate_per_residue_noise(x_data, eps=1e-3):
    x_start = x_data.x.contiguous()
    atoms = x_data.x[:, -4:].sum(dim=1)
    c_n_atoms = torch.where(atoms == 1)[0].to(x_start.device)
    p_atoms = torch.where(atoms == 0)[0].to(x_start.device)
    per_residue_noise = torch.rand(
        (c_n_atoms.shape[0]) // 4, x_start.shape[1], device=x_start.device
    )  # generate noise for each C4' atom
    per_residue_noise = torch.repeat_interleave(
        per_residue_noise, 4, dim=0
    )  # repeat it for all atoms in residue (except for P)
    noise = torch.zeros_like(x_start)
    noise[c_n_atoms] = per_residue_noise
    diff = torch.arange(0, len(p_atoms), device=x_start.device)
    relative_c4p = p_atoms - diff  # compute the index of each C4' for every P atom
    noise[p_atoms] = noise[
        c_n_atoms[relative_c4p]
    ]  # if there is a P atom, copy the noise from the corresponding C4' atom
    noise = noise + torch.randn_like(x_start, device=x_start.device) * eps

    return noise


class Sampler:
    def __init__(self, timesteps: int, channels: int = 3):
        self.timesteps = timesteps
        self.channels = channels
        # define beta schedule
        # self.betas = cosine_beta_schedule(timesteps=timesteps)
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @torch.no_grad()
    def p_sample(self, model, seqs, x_raw, t, t_index, coord_mask, atoms_mask):
        x = x_raw.x * coord_mask

        pred_noise = model(x_raw, seqs, t) * coord_mask

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        # Equation 12 in the DDIM paper
        # predict x_0
        pred_x0 = (
            x - sqrt_one_minus_alphas_cumprod_t * pred_noise
        ) / sqrt_alphas_cumprod_t

        alphas_cumprod_prev_t = self.extract(self.alphas_cumprod_prev, t, x.shape)

        # Equation 7 in DDIM paper
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alphas_cumprod_prev_t) * pred_noise

        x_prev = torch.sqrt(alphas_cumprod_prev_t) * pred_x0 + dir_xt

        x_raw.x = x_prev * coord_mask + x_raw.x * atoms_mask
        return x_raw.x

    def add_fixed(self, raw_x, fixed, t, t_index, x_start):
        if torch.any(fixed) and t_index > 0:
            denoised_raw = self.q_sample(x_start, t - 1)
            raw_x[fixed] = denoised_raw[fixed]
        if torch.any(fixed) and t_index == 0:
            raw_x[fixed] = x_start[fixed]
        return raw_x

    # Algorithm 2
    @torch.no_grad()
    def p_sample_loop(self, model, seqs, shape, context_mols, args=None):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        coord_mask = torch.ones_like(context_mols.x)
        coord_mask[:, 3:] = 0
        atoms_mask = 1 - coord_mask
        noise = torch.rand_like(context_mols.x, device=device)
        denoised = []

        context_mols.x = noise * coord_mask + context_mols.x * atoms_mask

        use_ddim = True  # Changed this to True to default to DDIM
        if use_ddim and args is not None and args.ddim_steps is not None:
            skip = self.timesteps // args.ddim_steps
            indices = list(range(0, self.timesteps, skip))[::-1]
        else:
            indices = list(range(self.timesteps))[::-1]
        print(f"Using {len(indices)} steps for sampling.")

        for i in tqdm(indices, desc="sampling loop time step", total=len(indices)):
            context_mols.x = self.p_sample(
                model,
                seqs,
                context_mols,
                torch.full((b,), i, device=device, dtype=torch.long),
                i,
                coord_mask,
                atoms_mask,
            )
            # denoised.append(context_mols.clone().cpu())
        denoised.append(context_mols.clone().cpu())
        return denoised

    @torch.no_grad()
    def sample(self, model, seqs, context_mols, args=None):
        return self.p_sample_loop(
            model,
            seqs,
            shape=context_mols.x.shape,
            context_mols=context_mols,
            args=args,
        )

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
