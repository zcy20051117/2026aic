from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def formal_optimizer_steps_per_epoch(
    *, sample_count: int, physical_batch_size: int, accumulation_steps: int
) -> int:
    """Compute the formal schedule length independently of diagnostic subsets."""
    if sample_count <= 0 or physical_batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("Sample, batch, and accumulation counts must be positive")
    microbatches = math.ceil(sample_count / physical_batch_size)
    return math.ceil(microbatches / accumulation_steps)


def build_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.AdamW:
    """Build exhaustive, disjoint decay/no-decay groups for trainable parameters."""
    owners = dict(model.named_modules())
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"Trainable parameter is shared or duplicated: {name}")
        seen.add(id(parameter))
        module_name, _, local_name = name.rpartition(".")
        owner = owners[module_name] if module_name else model
        lowered = name.lower()
        zero_decay = (
            local_name == "bias"
            or parameter.ndim <= 1
            or isinstance(owner, _NORM_TYPES)
            or isinstance(owner, nn.Embedding)
            or any(token in lowered for token in ("embedding", "embed", "query", "level_encoding"))
        )
        (no_decay if zero_decay else decay).append(parameter)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError("Optimizer grouping did not cover every trainable parameter exactly once")
    if not seen:
        raise ValueError("Model has no trainable parameters")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay), "group_name": "decay"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"})
    return torch.optim.AdamW(groups, lr=float(learning_rate), betas=betas)


class PolynomialWarmupScheduler:
    """Optimizer-step scheduler with linear warmup and polynomial decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        warmup_start_factor: float,
        power: float,
        min_learning_rate: float,
    ) -> None:
        if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
            raise ValueError("Require total_steps > warmup_steps >= 0")
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in (0, 1]")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.warmup_start_factor = float(warmup_start_factor)
        self.power = float(power)
        self.min_learning_rate = float(min_learning_rate)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_count = 0
        self._apply()

    def _lr(self, base_lr: float) -> float:
        if self.warmup_steps and self.step_count < self.warmup_steps:
            fraction = self.step_count / self.warmup_steps
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * fraction
            return base_lr * factor
        if self.step_count == self.warmup_steps:
            return max(self.min_learning_rate, base_lr)
        progress = min(
            1.0,
            (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps),
        )
        return max(self.min_learning_rate, base_lr * math.pow(1.0 - progress, self.power))

    def _apply(self) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = self._lr(base_lr)

    def step(self) -> None:
        self.step_count += 1
        self._apply()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def set_step_count(self, step_count: int) -> None:
        requested = int(step_count)
        if not 0 <= requested <= self.total_steps:
            raise ValueError(
                f"Scheduler step_count must be in [0, {self.total_steps}], got {requested}"
            )
        self.step_count = requested
        self._apply()

    def state_dict(self) -> dict[str, object]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_start_factor": self.warmup_start_factor,
            "power": self.power,
            "min_learning_rate": self.min_learning_rate,
            "base_lrs": self.base_lrs,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        structural = ("total_steps", "warmup_steps", "warmup_start_factor", "power", "min_learning_rate")
        for key in structural:
            if state[key] != getattr(self, key):
                raise ValueError(f"Scheduler mismatch for {key}")
        loaded_base_lrs = [float(value) for value in state["base_lrs"]]
        if loaded_base_lrs != self.base_lrs:
            raise ValueError("Scheduler base learning rates do not match")
        self.step_count = int(state["step_count"])
        self._apply()


@dataclass(frozen=True)
class AccumulationController:
    accumulation_steps: int
    total_microbatches: int

    def __post_init__(self) -> None:
        if self.accumulation_steps <= 0 or self.total_microbatches <= 0:
            raise ValueError("Accumulation and microbatch counts must be positive")

    def _window_bounds(self, microbatch_index: int) -> tuple[int, int]:
        if not 0 <= microbatch_index < self.total_microbatches:
            raise IndexError("microbatch_index is outside the epoch")
        start = (microbatch_index // self.accumulation_steps) * self.accumulation_steps
        end = min(start + self.accumulation_steps, self.total_microbatches)
        return start, end

    def window_size(self, microbatch_index: int) -> int:
        start, end = self._window_bounds(microbatch_index)
        return end - start

    def should_step(self, microbatch_index: int) -> bool:
        _, end = self._window_bounds(microbatch_index)
        return microbatch_index + 1 == end

    def scale_loss(self, loss: Tensor, microbatch_index: int) -> Tensor:
        return loss / self.window_size(microbatch_index)
