"""Codecs sit between user-facing data and the tensors a process works on.

``encode(data) -> Tensor`` and ``decode(Tensor) -> data``. Rewards and the
pipeline only ever see decoded data. A latent-space model plugs in here: write a
codec whose encode/decode run your VAE, and the rest of the stack is unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from .registry import CODECS, import_object


@CODECS.register("pixel")
class PixelCodec:
    """Images (B, C, H, W) / videos (B, C, T, H, W). Tensors live in [-1, 1], data is float32 numpy in [0, 1]."""

    def encode(self, data) -> torch.Tensor:
        return torch.as_tensor(np.asarray(data, dtype=np.float32)) * 2 - 1

    def decode(self, x: torch.Tensor) -> np.ndarray:
        return ((x.detach().float().clamp(-1, 1) + 1) / 2).cpu().numpy()


class ByteTokenizer:
    """UTF-8 bytes plus EOS (also padding) and MASK. Any object with these attributes and methods can replace it."""

    eos_id, mask_id, vocab_size = 256, 257, 258

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        out = bytearray()
        for i in ids:
            if i == self.eos_id:
                break
            out += b"_" if i == self.mask_id else bytes([i])
        return out.decode("utf-8", errors="replace")


@CODECS.register("text")
class TextCodec:
    """Strings <-> fixed-length token ids (B, L), padded with EOS."""

    def __init__(self, sample_shape, tokenizer: dict | None = None):
        (self.seq_len,) = sample_shape
        if tokenizer is None:
            self.tokenizer = ByteTokenizer()
        else:
            tokenizer = dict(tokenizer)
            self.tokenizer = import_object(tokenizer.pop("type"))(**tokenizer)
        self.mask_id = self.tokenizer.mask_id
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, texts: list[str]) -> torch.Tensor:
        x = torch.full((len(texts), self.seq_len), self.tokenizer.eos_id, dtype=torch.long)
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text)[: self.seq_len]
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return x

    def decode(self, x: torch.Tensor) -> list[str]:
        return [self.tokenizer.decode(row) for row in x.tolist()]
