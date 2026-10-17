"""omnidiff.distill: few-step distillation for text, image and video diffusion.

Methods (``DISTILLERS`` registry): kd, dmd2, consistency, flow_map, sdtt. See docs/scope/distillation.md.
"""

from . import methods  # noqa: F401
from .trainer import DistillConfig, DistillTrainer
