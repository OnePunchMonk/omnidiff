"""diffusion_omni.distill: few-step distillation for text, image and video diffusion.

Methods (``DISTILLERS`` registry): kd, dmd2, dmd2_reward, consistency, flow_map, rcm, progressive, sdtt, coupled_kd, dcd. See docs/scope/distillation.md.
"""

from . import methods  # noqa: F401
from .trainer import DistillConfig, DistillTrainer
