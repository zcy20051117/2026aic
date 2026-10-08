from __future__ import annotations

import pytest

from aic_dino_m2f.train import _resolve_end_epoch


def test_end_epoch_limits_initialization_run_to_ten_epochs() -> None:
    assert _resolve_end_epoch(start_epoch=35, configured_max_epochs=150, requested_end_epoch=44) == 44


def test_end_epoch_rejects_a_range_before_the_initialization_epoch() -> None:
    with pytest.raises(ValueError, match="before start epoch"):
        _resolve_end_epoch(start_epoch=35, configured_max_epochs=150, requested_end_epoch=34)
