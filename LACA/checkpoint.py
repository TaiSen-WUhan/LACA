"""Resolve public checkpoint names and load the saved model configuration."""

import pickle
import types
from pathlib import Path

import torch

from .config import CfgNode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "sf": "laca.pth",
    "laca": "laca.pth",
    "l_sf": "l-laca.pth",
    "l-laca": "l-laca.pth",
    "laca-l": "l-laca.pth",
    "kitti": "kitti.pth",
}


class _ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Released weights use the lowercase package name; keep their bytes intact.
        if name == "CfgNode" and module in ("laca.config.config", "LACA.config.config"):
            return CfgNode
        return super().find_class(module, name)


_checkpoint_pickle = types.ModuleType("laca_checkpoint_pickle")
_checkpoint_pickle.__dict__.update(vars(pickle))
_checkpoint_pickle.Unpickler = _ConfigUnpickler


def resolve_checkpoint(name_or_path):
    """Accept a local path or sf/l_sf/kitti, independently of the working directory."""
    path = Path(name_or_path).expanduser()
    candidates = [path, PROJECT_ROOT / path]
    if str(name_or_path) in CHECKPOINTS:
        candidates.append(PROJECT_ROOT / "weight" / CHECKPOINTS[str(name_or_path)])
    elif path.name == str(path):
        candidates.append(PROJECT_ROOT / "weight" / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Checkpoint not found: {name_or_path}. "
        f"Use a file path or one of {', '.join(CHECKPOINTS)} in {PROJECT_ROOT / 'weight'}."
    )


def load_checkpoint(name_or_path):
    """Load trusted local checkpoints, including their serialized CfgNode."""
    checkpoint = torch.load(
        resolve_checkpoint(name_or_path), map_location="cpu", weights_only=False,
        pickle_module=_checkpoint_pickle,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("A LACA checkpoint must contain a 'model' state dictionary.")
    return checkpoint
