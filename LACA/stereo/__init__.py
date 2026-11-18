# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .NMRF import build_criterion, NMRF


def build_model(cfg, pretrained=False):
    return NMRF(cfg, backbone_pretrained=pretrained)