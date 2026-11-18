"""DDHS sequence expansion and restoration; no learned parameters."""
import torch
from torch import nn
from einops import rearrange


class DDHSScan(nn.Module):
    """Use row/column traversals and their reverses for each disparity proposal."""

    def scan(self, x):
        if x.ndim != 5:
            raise ValueError("DDHS expects [B,C,N,H,W].")
        columns = rearrange(x, 'b c n h w -> b c (n w h)')
        rows = rearrange(x, 'b c n h w -> b c (n h w)')
        return torch.stack((columns, columns.flip(-1), rows, rows.flip(-1)), dim=1)

    def restore(self, sequences, batch, height, width, proposals):
        if sequences.ndim != 4 or sequences.shape[:2] != (batch, 4):
            raise ValueError("DDHS restoration expects [B,4,C,N*H*W].")
        if sequences.shape[-1] != proposals * height * width:
            raise ValueError("DDHS sequence length does not match the cost volume.")
        # Undo each traversal before summing the four scan outputs.
        columns = sequences[:, 0] + sequences[:, 1].flip(-1)
        rows = sequences[:, 2] + sequences[:, 3].flip(-1)
        columns = rearrange(columns, 'b c (n w h) -> b h w n c', n=proposals, h=height, w=width)
        rows = rearrange(rows, 'b c (n h w) -> b h w n c', n=proposals, h=height, w=width)
        return columns + rows

    def forward(self, x):
        return self.scan(x)
