"""Combine checkpoint architecture, experiment YAML, and command-line overrides."""
from pathlib import Path

from .config import get_cfg


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_config(config_file="", opts=None, model_config=None, data_root=None):
    cfg = get_cfg()
    if model_config is not None:
        for name in ("BACKBONE", "DPN", "NMP", "LACA"):
            cfg[name] = model_config[name].clone()
    if config_file:
        path = Path(config_file).expanduser()
        if not path.is_file():
            path = PROJECT_ROOT / path
        cfg.merge_from_file(str(path))
    cfg.merge_from_list(opts or [])
    if data_root is not None:
        cfg.DATASETS.ROOT = str(data_root)
    root = Path(cfg.DATASETS.ROOT).expanduser()
    cfg.DATASETS.ROOT = str(root if root.is_absolute() else PROJECT_ROOT / root)
    cfg.freeze()
    return cfg
