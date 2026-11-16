"""CLIP image/text embeddings with OpenCLIP, running locally (CUDA if available)."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
_FALLBACK_MEAN = (0.48145466, 0.4578275, 0.40821073)  # OpenAI CLIP normalisation
_FALLBACK_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipEmbedder:
    """Lazily loads an OpenCLIP model and turns frames or prompts into unit vectors.

    The first use downloads the weights (~600 MB) into the Hugging Face cache;
    after that everything is offline.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._device = device
        self.batch_size = batch_size

    @cached_property
    def _bundle(self):  # type: ignore[no-untyped-def]
        import open_clip
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=device
        )
        model.eval()
        if device == "cuda":
            model = model.half()  # half the VRAM, faster; negligible accuracy change
        cfg = getattr(model.visual, "preprocess_cfg", {}) or {}
        mean = torch.tensor(cfg.get("mean", _FALLBACK_MEAN), device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get("std", _FALLBACK_STD), device=device).view(1, 3, 1, 1)
        tokenizer = open_clip.get_tokenizer(self.model_name)
        return torch, model, tokenizer, device, mean, std

    @property
    def device(self) -> str:
        """Device the model runs on ("cuda" or "cpu")."""
        return self._bundle[3]

    def images(self, frames: np.ndarray) -> np.ndarray:
        """Embed uint8 frames (N, H, W, 3) into L2-normalised float32 vectors (N, D)."""
        torch, model, _, device, mean, std = self._bundle
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(frames[i : i + self.batch_size]))
                x = batch.to(device).permute(0, 3, 1, 2).float().div(255.0)
                x = (x - mean) / std
                if device == "cuda":
                    x = x.half()
                emb = model.encode_image(x).float()
                out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)

    def texts(self, prompts: Sequence[str]) -> np.ndarray:
        """Embed text prompts into L2-normalised float32 vectors (N, D)."""
        torch, model, tokenizer, device, _, _ = self._bundle
        with torch.no_grad():
            emb = model.encode_text(tokenizer(list(prompts)).to(device)).float()
            return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
