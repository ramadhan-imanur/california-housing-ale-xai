"""Unit tests for Accumulated Local Effects (ALE) and Partial Dependence Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from ale_engine import ALEEngine, ALE1DResult, ALE2DResult, PDP1DResult


@pytest.fixture
def synthetic_additive_data():
    """Generate synthetic bivariate normal data with known linear additive relationship."""
    np.random.seed(42)
    n = 2000
    cov = [[1.0, 0.95], [0.95, 1.0]]
    X_mat = np.random.multivariate_normal([0, 0], cov, size=n)
    df = pd.DataFrame(X_mat, columns=["x1", "x2"])
    # Additive function with no interaction: y = 2*x1 + 3*x2 + 5
    y = 2.0 * df["x1"] + 3.0 * df["x2"] + 5.0
    model = LinearRegression().fit(df, y)
    return df, y, model


def test_ale_1d_additive_recovery(synthetic_additive_data):
    """Test that 1D ALE perfectly recovers true main effect slope on an additive model."""
    df, _, model = synthetic_additive_data
    res = ALEEngine.compute_1d(model.predict, df, "x1", grid_size=30)

    assert isinstance(res, ALE1DResult)
    assert len(res.grid_values) > 2
    assert len(res.ale_values) == len(res.grid_values)

    # Slope of ALE on x1 should be exactly 2.0
    slope, intercept = np.polyfit(res.grid_values, res.ale_values, 1)
    np.testing.assert_allclose(slope, 2.0, rtol=1e-5, atol=1e-5)


def test_ale_1d_zero_mean_centering(synthetic_additive_data):
    """Test that 1D ALE satisfies zero-mean expectation over empirical sample distribution."""
    df, _, model = synthetic_additive_data
    res = ALEEngine.compute_1d(model.predict, df, "x1", grid_size=40)

    # Linear interpolation at empirical sample values
    sample_effects = np.interp(res.sample_values, res.grid_values, res.ale_values)
    mean_effect = float(np.mean(sample_effects))

    assert abs(mean_effect) < 1e-10


def test_ale_2d_additive_model_zero_interaction(synthetic_additive_data):
    """Test that 2D ALE yields zero interaction effect on a pure additive model."""
    df, _, model = synthetic_additive_data
    res = ALEEngine.compute_2d(model.predict, df, "x1", "x2", grid_size=15, min_samples=1)

    assert isinstance(res, ALE2DResult)
    # On an additive model, all valid interaction values should be identically 0
    valid_cells = ~np.isnan(res.ale_grid)
    np.testing.assert_allclose(res.ale_grid[valid_cells], 0.0, atol=1e-10)


def test_ale_2d_two_way_marginal_centering():
    """Test that 2D ALE two-way centering removes marginal main effects across rows and cols."""
    np.random.seed(42)
    n = 1500
    df = pd.DataFrame({
        "x1": np.random.uniform(-2, 2, n),
        "x2": np.random.uniform(-2, 2, n),
    })
    # Function with main effects and non-linear interaction: y = 3*x1 + 2*x2 + 4*(x1*x2)
    y = 3.0 * df["x1"] + 2.0 * df["x2"] + 4.0 * (df["x1"] * df["x2"])
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=100, random_state=42).fit(df, y)

    res = ALEEngine.compute_2d(model.predict, df, "x1", "x2", grid_size=15, min_samples=1)

    # Check that row-wise and column-wise weighted marginal sums of interaction are near zero
    counts = res.cell_counts.astype(float)
    grid = np.nan_to_num(res.ale_grid, nan=0.0)

    for i in range(res.cell_counts.shape[0]):
        row_weight = np.sum(counts[i, :])
        if row_weight > 0:
            row_mean = np.sum(counts[i, :] * grid[i, :]) / row_weight
            assert abs(row_mean) < 1e-4

    for j in range(res.cell_counts.shape[1]):
        col_weight = np.sum(counts[:, j])
        if col_weight > 0:
            col_mean = np.sum(counts[:, j] * grid[:, j]) / col_weight
            assert abs(col_mean) < 1e-4


def test_pdp_1d_consistent_centering(synthetic_additive_data):
    """Test that PDP 1D with distribution-weighted centering satisfies zero-mean expectation."""
    df, _, model = synthetic_additive_data
    res = ALEEngine.compute_pdp_1d(model.predict, df, "x1", grid_size=30)

    assert isinstance(res, PDP1DResult)
    sample_effects = np.interp(df["x1"].to_numpy(), res.grid_values, res.pdp_centered)
    mean_effect = float(np.mean(sample_effects))

    assert abs(mean_effect) < 1e-10
