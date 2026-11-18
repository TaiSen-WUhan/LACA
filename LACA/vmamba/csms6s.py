"""Selective scan with an optional Oflex CUDA kernel and a PyTorch fallback."""
import torch
import torch.nn.functional as F

try:
    import selective_scan_cuda_oflex as _oflex
except ImportError:
    _oflex = None


def selective_scan_torch(u, delta, A, B, C, D=None, delta_bias=None):
    """Differentiable reference recurrence for CPU and environments without Oflex."""
    batch, groups, states, length = B.shape
    channels = u.shape[1] // groups
    u, delta, A, B, C = (value.float() for value in (u, delta, A, B, C))
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    delta = F.softplus(delta)
    B = B[:, :, None].expand(-1, -1, channels, -1, -1).reshape(batch, -1, states, length)
    C = C[:, :, None].expand(-1, -1, channels, -1, -1).reshape(batch, -1, states, length)
    state = u.new_zeros(batch, groups * channels, states)
    outputs = []
    for index in range(length):
        transition = torch.exp(delta[:, :, index, None] * A[None])
        state = transition * state + delta[:, :, index, None] * B[:, :, :, index] * u[:, :, index, None]
        outputs.append((state * C[:, :, :, index]).sum(-1))
    output = torch.stack(outputs, dim=-1)
    return output if D is None else output + u * D.float()[None, :, None]


class _SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias):
        output, state, *_ = _oflex.fwd(u, delta, A, B, C, D, delta_bias, True, 1, True)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        u, delta, A, B, C, D, delta_bias, state = ctx.saved_tensors
        gradients = _oflex.bwd(u, delta, A, B, C, D, delta_bias, grad_output.contiguous(), state, True, 1)
        return tuple(gradients[:7])


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None):
    # CPU never dispatches to a CUDA extension, even when that extension is installed.
    if u.is_cuda and _oflex is not None:
        return _SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias)
    return selective_scan_torch(u, delta, A, B, C, D, delta_bias)
