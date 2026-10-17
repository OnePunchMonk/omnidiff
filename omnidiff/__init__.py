"""omnidiff: RL post-training and inference for text, image and video diffusion, in PyTorch + NumPy."""

# Importing the component modules registers the built-ins.
from . import codecs, datasets, prompts, rewards  # noqa: F401
from .nn import adapters, backbones, conditioners, denoiser  # noqa: F401
from .processes import flow, gaussian, masked  # noqa: F401
from .rl import algorithms  # noqa: F401

from .build import Components, build
from .config import load_config
from .pipeline import Generation, Pipeline
from .rl.trainer import RLTrainer
from .sampling import Trajectory, sample
from .sft import SFTTrainer

__version__ = "0.1.0"
