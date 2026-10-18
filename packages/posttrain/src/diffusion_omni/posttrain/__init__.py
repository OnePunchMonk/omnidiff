"""diffusion_omni.posttrain: SFT and online RL (GRPO, DDPO) on the denoising MDP."""

from .rl import algorithms  # noqa: F401
from .rl.trainer import RLConfig, RLTrainer
from .sft import SFTConfig, SFTTrainer
