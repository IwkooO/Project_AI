import inspect
from pathlib import Path

import torch


def torch_load_checkpoint(path: Path, map_location):
    """
    Torch checkpoint loader compatible across torch versions.

    - torch>=2.6 defaults weights_only=True, which breaks loading checkpoints that store
      non-tensor metadata (e.g., numpy scalars). We explicitly set weights_only=False when supported.
    - torch<2.0 does not support the weights_only argument.
    """
    params = inspect.signature(torch.load).parameters
    if "weights_only" in params:
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)


