"""The SS3D path used by the released LACA and LACA-L checkpoints."""
import math
import torch
from torch import nn
from einops import rearrange

from .ddhs import DDHSScan
from .csms6s import selective_scan_fn


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class SS3D(nn.Module):
    """Scan a [B,H,W,N,C] cost volume in four spatial directions."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = self.d_inner = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.k_group = 4
        self.ddhs = DDHSScan()
        self.out_norm = nn.LayerNorm(d_model)
        # Keep both projection halves to match the released checkpoint layout.
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.act = nn.SiLU()
        self.conv3d = nn.Conv3d(d_model, d_model, 3, padding=1, groups=d_model)
        projections = [nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
                       for _ in range(self.k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in projections]))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, self.dt_rank, d_model, 1.0, "random", 0.001, 0.1, 1e-4, k_group=4
        )

    def forward(self, x):
        x = self.in_proj(x).chunk(2, dim=-1)[0]
        x = rearrange(x, 'b h w n c -> b c n h w')
        x = self.act(self.conv3d(x))
        batch, channels, proposals, height, width = x.shape
        length = proposals * height * width
        sequences = self.ddhs.scan(x)
        projected = torch.einsum('b k d l, k c d -> b k c l', sequences, self.x_proj_weight)
        delta, state_b, state_c = projected.split([self.dt_rank, self.d_state, self.d_state], dim=2)
        delta = torch.einsum('b k r l, k d r -> b k d l', delta, self.dt_projs_weight)
        # State-space recurrence is evaluated in float32 for numerical stability.
        scanned = selective_scan_fn(
            sequences.reshape(batch, -1, length).float(),
            delta.contiguous().reshape(batch, -1, length).float(),
            -self.A_logs.float().exp(), state_b.contiguous().float(), state_c.contiguous().float(),
            self.Ds.float(), self.dt_projs_bias.reshape(-1).float(),
        ).reshape(batch, 4, channels, length)
        restored = self.ddhs.restore(scanned, batch, height, width, proposals)
        return self.out_proj(self.out_norm(restored).to(x.dtype))


class VMM3D(nn.Module):
    """Pre-normalized residual SS3D block."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.op = SS3D(hidden_dim)

    def forward(self, x):
        return x + self.op(self.norm(x))
