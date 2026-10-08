from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass
class EarlyStoppingState:
    patience: int
    best_score: float = -math.inf
    best_epoch: int = -1
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def update(self, score: float, *, epoch: int) -> bool:
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Validation score must be finite")
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    def state_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "EarlyStoppingState":
        return cls(
            patience=int(state["patience"]),
            best_score=float(state["best_score"]),
            best_epoch=int(state["best_epoch"]),
            bad_epochs=int(state["bad_epochs"]),
        )
