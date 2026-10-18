from __future__ import annotations

import torch
import torch.nn.functional as F

from ..registry import PROCESSES
from .base import DiffusionProcess, Transition, reduce


@PROCESSES.register("masked")
class MaskedDiffusion(DiffusionProcess):
    """Absorbing-state discrete diffusion (MDLM / LLaDA style), linear schedule.

    Each free token is masked with probability t. The model predicts logits over
    the clean token at every position. A reverse step from t to s unmasks each
    masked token with probability (t - s) / t and draws it from the prediction;
    decoded tokens are carried over unchanged. So the step's log-prob is exact:

        log pi = sum_{unmasked now}  [log((t - s) / t) + log p(token)]
               + sum_{still masked}  log(s / t)

    Prompts and infilling constraints are pre-filled positions: ``cond["prefix"]``
    is (B, L) with a token id where the position is fixed and -1 where it is free.

    ``sampler: coupled`` runs the same ancestral process as a deterministic function of
    per-trajectory noise drawn once up front: an unmask time per position (decoded on the
    step whose interval contains it, which happens with probability (t - s) / t) and Gumbel
    noise per position and token (argmax(log p + g) is a categorical sample). With
    ``coupled_gumbel: 1`` the output distribution is exactly the ancestral one. Noise-conditioned
    students read ``cond["token_noise"]`` themselves and sample with ``coupled_gumbel: 0``.

    ``coupled_noise: uniform`` replaces the (B, L, V) Gumbel noise with one uniform per position and
    draws the token by inverting the CDF of p (in token-id order). It is just as exact, and the noise
    stays O(L) for real vocabularies. ``coupled_gumbel: 0`` again means argmax.
    """

    sampler = "ancestral"  # for masked diffusion, "predict x0 then re-mask" is exactly the ancestral step

    def __init__(self, mask_id: int, temperature: float = 1.0, reduction: str = "sum", t_min: float = 1e-3,
                 sampler: str = "ancestral", vocab_size: int | None = None, coupled_gumbel: float = 1.0,
                 coupled_noise: str = "gumbel"):
        if sampler not in ("ancestral", "renoise", "coupled"):
            raise ValueError(f"unknown sampler {sampler!r}")
        if coupled_noise not in ("gumbel", "uniform"):
            raise ValueError(f"coupled_noise must be gumbel|uniform, got {coupled_noise!r}")
        self.mask_id, self.temperature, self.reduction, self.t_min = mask_id, temperature, reduction, t_min
        self.sampler = "ancestral" if sampler == "renoise" else sampler
        self.vocab_size, self.coupled_gumbel, self.coupled_noise = vocab_size, coupled_gumbel, coupled_noise

    def init_noise(self, shape, device, generator=None) -> dict:
        """Noise that makes coupled sampling deterministic: ``unmask_time`` (B, L) and ``token_noise``,
        (B, L, V) Gumbel or (B, L) uniform."""
        unif = lambda *s: torch.rand(s, device=device, generator=generator)  # noqa: E731
        if self.coupled_noise == "uniform":
            return {"unmask_time": unif(*shape), "token_noise": unif(*shape)}
        if self.vocab_size is None:
            raise ValueError("coupled sampling needs vocab_size")
        gumbel = -torch.log(-torch.log(unif(*shape, self.vocab_size).clamp(1e-20, 1 - 1e-7)))
        return {"unmask_time": unif(*shape), "token_noise": gumbel}

    def q_sample(self, x0, t, generator=None, maskable=None):
        """Mask each position with probability t (scalar or (B,)); positions outside ``maskable`` are kept."""
        t = torch.as_tensor(t, device=x0.device, dtype=torch.float32)
        t = t.view(-1, 1) if t.dim() else t
        masked = torch.rand(x0.shape, device=x0.device, generator=generator) < t
        if maskable is not None:
            masked = masked & maskable
        return torch.where(masked, self.mask_id, x0)

    def log_probs(self, logits):
        """Per-position log-probabilities over clean tokens, with the mask token excluded."""
        return self._log_probs(logits)

    def timesteps(self, num_steps, device):
        return torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    def prior(self, shape, cond, device, generator=None):
        x = torch.full(shape, self.mask_id, dtype=torch.long, device=device)
        prefix = (cond or {}).get("prefix")
        if prefix is not None:
            x = torch.where(prefix >= 0, prefix.to(device), x)
        return x

    def training_loss(self, model, x0, cond, drop_cond=None):
        B = x0.shape[0]
        t = torch.rand(B, device=x0.device) * (1 - self.t_min) + self.t_min
        maskable = (cond or {}).get("loss_mask")
        maskable = torch.ones_like(x0, dtype=torch.bool) if maskable is None else maskable.bool()
        x_t = self.q_sample(x0, t, maskable=maskable)
        masked = x_t == self.mask_id
        logits = model(x_t, t, cond, drop_cond)
        nll = F.cross_entropy(logits.float().transpose(1, 2), x0, reduction="none")
        # MDLM's ELBO weight for the linear schedule is 1 / t.
        return (nll * masked / t[:, None]).sum() / maskable.sum().clamp(min=1)

    def _log_probs(self, logits):
        logits = logits.float() / self.temperature
        logits = logits.index_fill(-1, torch.tensor([self.mask_id], device=logits.device), float("-inf"))
        return logits.log_softmax(-1)

    def transition(self, pred, x_t, t, t_next, x_next=None, generator=None, cond=None):
        if self.sampler == "coupled":
            return self._coupled_transition(pred, x_t, t, t_next, cond, x_next)
        logp = self._log_probs(pred)
        is_masked = x_t == self.mask_id
        p = ((t - t_next) / t).clamp(0, 1)
        if x_next is None:
            u = torch.rand(x_t.shape, device=x_t.device, generator=generator)
            draws = torch.multinomial(logp.exp().flatten(0, -2), 1, generator=generator).view_as(x_t)
            x_next = torch.where(is_masked & (u < p), draws, x_t)
        unmasked = is_masked & (x_next != self.mask_id)
        stayed = is_masked & ~unmasked
        tok_logp = logp.gather(-1, torch.where(unmasked, x_next, 0).unsqueeze(-1)).squeeze(-1)
        per_tok = torch.where(unmasked, tok_logp + p.clamp(min=1e-12).log(), torch.zeros_like(tok_logp))
        per_tok = per_tok + stayed * (1 - p).clamp(min=1e-12).log()
        stats = {"logp": logp, "is_masked": is_masked, "p": p}
        return Transition(x_next, reduce(per_tok, self.reduction), stats)

    def _coupled_transition(self, pred, x_t, t, t_next, cond, x_next=None):
        logp = self._log_probs(pred)
        if x_next is None:
            tau = cond["unmask_time"]
            decode = (x_t == self.mask_id) & (tau >= t_next) & (tau < t)
            if self.coupled_gumbel == 0:
                tokens = logp.argmax(-1)
            elif self.coupled_noise == "uniform":  # inverse CDF; the first token whose CDF exceeds u has p > 0
                cdf = logp.exp().cumsum(-1)
                u = cond["token_noise"].unsqueeze(-1) * cdf[..., -1:]
                tokens = torch.searchsorted(cdf, u, right=True).squeeze(-1).clamp(max=cdf.shape[-1] - 1)
            else:
                tokens = (logp + self.coupled_gumbel * cond["token_noise"]).argmax(-1)
            x_next = torch.where(decode, tokens, x_t)
        return Transition(x_next, None, {"logp": logp})

    def kl(self, new, ref):
        # Both policies share the unmask probability, so the step KL is the categorical
        # KL at each masked position weighted by that probability.
        lp, lq = new.stats["logp"], ref.stats["logp"]
        diff = (lp - lq).masked_fill(~lp.isfinite(), 0.0)
        per_pos = (lp.exp() * diff).sum(-1) * new.stats["is_masked"] * new.stats["p"]
        return reduce(per_pos, self.reduction)
