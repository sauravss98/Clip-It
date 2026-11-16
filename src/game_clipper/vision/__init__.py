"""Optional local vision stage: CLIP frame embeddings, scene filtering and a trainable model.

Everything here runs offline on the user's own GPU (or CPU). It needs the optional
extra: ``pip install "game-clipper[vision]"``. Modules that import torch do so lazily,
so the rest of the package works without it.
"""

from __future__ import annotations

from importlib.util import find_spec

INSTALL_HINT = (
    'Vision features need the optional extra: pip install -e ".[vision]"\n'
    "For NVIDIA GPUs install the CUDA build of PyTorch first, e.g.:\n"
    "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
)


def available() -> bool:
    """True when torch and open_clip are installed (without importing them)."""
    return find_spec("torch") is not None and find_spec("open_clip") is not None
