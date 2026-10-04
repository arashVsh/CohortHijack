from __future__ import annotations

import numpy as np

from scripts.reviewer_statistics import bh_adjust, bootstrap_ci


def test_bh_adjust_preserves_order_and_monotonicity() -> None:
    adjusted = bh_adjust([0.04, 0.001, 0.03, 0.2])
    np.testing.assert_allclose(adjusted, [0.05333333333333334, 0.004, 0.05333333333333334, 0.2])


def test_bootstrap_degenerate_binary_rate_is_exactly_zero() -> None:
    estimate, low, high = bootstrap_ci(
        np.zeros(20), n_boot=100, rng=np.random.default_rng(1)
    )
    assert (estimate, low, high) == (0.0, 0.0, 0.0)
