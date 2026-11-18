import torch
from torch import nn
import torch.nn.functional as F

from ..config import configurable
from .NMP import Head, Propagation, PropagationLayer
from .submodule import init_weights


class DPN(nn.Module):
    """Disparity proposal extraction network.

    Args:
        cost_group (int): group number of groupwise cost volume
        num_proposals (int): number of proposals for each pixel
        feat_dim (int): dimension of backbone feature map
        context_dim (int): dimension of visual context
        prop_embed_dim (int): dimension of label seed embedding
        split_size (int): width of stripe
        prop_n_heads: head of attention
    """
    @configurable
    def __init__(self, cost_group, num_proposals, feat_dim, prior_dim, num_prop_layers,
                 prop_embed_dim, mlp_ratio,  split_size, prop_n_heads, activation="gelu",
                 attn_drop=0.,  proj_drop=0., drop_path=0., dropout=0.,
    ):
        super().__init__()

        # 1D convolutions sliding along the disparity dimension.
        # Intuition: high-pass filter to make the disparity modal prominent
        self.mlp = nn.Sequential(
            nn.Conv1d(cost_group, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=5, stride=1, padding=2),
        )
        self.eps = 1e-3
        self.num_proposals = num_proposals
        self.cost_group = cost_group

        # ---- label seed propagation ---- #
        # to obtain visual context
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, prop_embed_dim // 2, 1, 1, 0, bias=False))
        self.prior_proj = nn.Linear(prior_dim, prop_embed_dim // 2)

        prop_layer = PropagationLayer(prop_embed_dim, mlp_ratio=mlp_ratio, context_dim=prop_embed_dim//2,
                                      split_size=split_size, n_heads=prop_n_heads, activation=activation, 
                                      attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, dropout=dropout)
        prop_norm = nn.LayerNorm(prop_embed_dim)
        self.propagation = Propagation(prop_embed_dim, cost_group, prop_layer=prop_layer, num_layers=num_prop_layers,
                                       norm=prop_norm)

        self.prop_head = Head(prop_embed_dim, prop_embed_dim, 1, 3)

        self.apply(init_weights)
        nn.init.constant_(self.prop_head.layers[-1].weight.data, 0.)
        nn.init.constant_(self.prop_head.layers[-1].bias.data, 0.)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_proposals": cfg.DPN.NUM_PROPOSALS,
            "cost_group": cfg.DPN.COST_GROUP,
            "feat_dim": cfg.BACKBONE.OUT_CHANNELS,
            "prior_dim": cfg.NMP.INFER_EMBED_DIM,
            "num_prop_layers": cfg.NMP.NUM_PROP_LAYERS,
            "prop_embed_dim": cfg.NMP.PROP_EMBED_DIM,
            "mlp_ratio": cfg.NMP.MLP_RATIO,
            "split_size": cfg.NMP.SPLIT_SIZE,
            "prop_n_heads": cfg.NMP.PROP_N_HEADS,
            "attn_drop": cfg.NMP.ATTN_DROP,
            "proj_drop": cfg.NMP.PROJ_DROP,
            "drop_path": cfg.NMP.DROP_PATH,
            "dropout": cfg.NMP.DROPOUT,
        }

    def forward(self, cost_volume, context, depth_prior):
        """
        cost_volume: [B,G,D,H,W]
        context:     [B,C,H,W]
        depth_prior:   [B,H,W,C']
        returns:
            prob: [BHW,D], initial disparity probability
            labels: [BHW,num_proposals], disparity candidates
        """
        # ---- step 1: extract disparity modals as label seeds ---- #
        cost_volume = cost_volume.permute(0, 3, 4, 1, 2).flatten(0, 2)  # [BHW,G,D]
        cost = self.mlp(cost_volume).squeeze(1)  # [BHW,D]
        prob = F.softmax(cost, dim=-1)
        out = F.max_pool1d(prob.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        non_local_max = (prob != out) & (prob > self.eps)

        prob_ = prob.clone().detach()
        prob_[non_local_max] = self.eps
        _, disp_seeds = torch.topk(prob_, self.num_proposals, dim=-1)

        # ---- step 2: label seed propagation ---- #
        context = self.proj(context)  # visual context is used in affinity computation
        context = context.permute(0, 2, 3, 1).contiguous()
        prior_proj = self.prior_proj(depth_prior)
        tgt, disp_seeds = self.propagation(cost_volume, disp_seeds, context, prior_proj)
        update = self.prop_head(tgt).view(*disp_seeds.shape)
        proposals = F.relu(update + disp_seeds)  # disparity proposals

        return prob, proposals