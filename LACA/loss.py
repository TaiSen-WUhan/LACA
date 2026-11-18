import torch
from torch import nn

from .stereo import build_criterion as build_stereo_criterion


def invalid_to_nans(arr, valid_mask):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    return arr

def compute_scale_and_shift(input, mask):
    """
    input, mask: [B,H,W]
    Returns:
        bias, scale: [B]
    """
    nan_input = invalid_to_nans(input, mask).flatten(1)
    bias = torch.nanquantile(nan_input, 0.5, dim=1)
    scale = torch.abs(nan_input - bias[:, None]).nanmean(dim=1) + 1e-6
    bias[bias.isnan()] = 0
    scale[scale.isnan()] = 1
    return bias, scale

def normalize_disparity(disp, mask):
    """
    disp, mask: [B,H,W]
    """
    bias, scale = compute_scale_and_shift(disp, mask)
    return (disp - bias[:, None, None]) / scale[:, None, None]

def sequence_affine_invariant_loss(predictions, target, mask, loss_gamma=0.9):
    """
    predictions: [N,B,H,W]
    target:      [B,H,W]
    mask:        [B,H,W]
    """
    n_predictions = predictions.shape[0]
    
    # We adjust the loss_gamma so it is consistent for any number of recurrent iterations
    adjust_loss_gamma = loss_gamma**(10/max(n_predictions-1, 1))
    loss = 0.0
    target = normalize_disparity(target, mask)
    
    for i in range(n_predictions):
        i_weight = adjust_loss_gamma**(n_predictions - i - 1)
        i_pred = normalize_disparity(predictions[i], mask)
        i_loss = torch.nan_to_num((i_pred - target).abs(), posinf=0) * mask
        i_loss = i_loss.flatten(1).sum(dim=1) / (mask.flatten(1).sum(dim=1) + 1e-6)
        loss += i_weight * i_loss.mean()
    
    assert not torch.isnan(loss)
    return loss
        
        
class LACACriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.stereo_criterion = build_stereo_criterion(cfg)
        self.loss_gamma = cfg.LACA.RECURRENT_LOSS_GAMMA
        self.weight_dict = dict(self.stereo_criterion.weight_dict)
        self.weight_dict.update({'loss_recurrent': cfg.LACA.RECURRENT_LOSS_WEIGHT})
        
    def forward(self, outputs, targets, log):
        loss_dict = self.stereo_criterion(outputs, targets, log)
        
        state_predictions = outputs['mono_prediction']
        device = state_predictions.device
        tgt_disp = targets['disp'].to(device)
        valid = targets['valid'].to(device).bool()
        valid = valid & (tgt_disp > 0) & (tgt_disp < self.stereo_criterion.max_disp)
        loss_recurrent = sequence_affine_invariant_loss(state_predictions, tgt_disp, valid, self.loss_gamma)
        
        loss_dict.update({'loss_recurrent': loss_recurrent})
        return loss_dict
        
        
def build_criterion(cfg):
    return LACACriterion(cfg)