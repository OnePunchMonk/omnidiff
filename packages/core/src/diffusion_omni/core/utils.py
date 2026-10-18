from __future__ import annotations

import json
import random
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from .config import save_config

WEIGHTS, CONFIG = "model.pt", "config.yaml"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str | None = "auto") -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_generator(seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def index_cond(cond: dict, idx) -> dict:
    return {k: v[idx] for k, v in cond.items()}


def save_checkpoint(path, state_dict: dict, cfg: dict) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path / WEIGHTS)
    save_config(cfg, path / CONFIG)
    return path


def load_weights(path, map_location="cpu") -> dict:
    path = Path(path)
    return torch.load(path / WEIGHTS if path.is_dir() else path, map_location=map_location, weights_only=True)


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, step: int, metrics: dict) -> None:
        row = {"step": step, "time": round(time.time() - self.start, 2), **{k: float(v) for k, v in metrics.items()}}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)


def to_grid(samples: np.ndarray, ncol: int | None = None, pad: int = 1) -> np.ndarray:
    """(B, C, H, W) images -> grid; (B, C, T, H, W) videos -> one row per video. Returns HWC uint8."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 5:
        B, C, T, H, W = x.shape
        tiles, ncol = x.transpose(0, 2, 1, 3, 4).reshape(B * T, C, H, W), T
    else:
        tiles = x
        ncol = ncol or int(np.ceil(np.sqrt(len(x))))
    N, C, H, W = tiles.shape
    nrow = -(-N // ncol)
    grid = np.ones((C, nrow * (H + pad) + pad, ncol * (W + pad) + pad), dtype=np.float32)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, ncol)
        grid[:, pad + r * (H + pad): pad + r * (H + pad) + H, pad + c * (W + pad): pad + c * (W + pad) + W] = tile
    return (grid.transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)


def write_png(path, img: np.ndarray, scale: int = 1) -> None:
    """Minimal PNG writer (numpy + zlib), HWC uint8 with 1 or 3 channels."""
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    h, w, c = img.shape
    color_type = {1: 0, 3: 2}[c]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
