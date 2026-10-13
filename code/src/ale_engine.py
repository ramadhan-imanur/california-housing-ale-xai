"""Accumulated Local Effects (ALE) and Partial Dependence Engine.

This module implements first-order (1D) and second-order (2D) Accumulated Local
Effects (ALE) alongside Partial Dependence Plots (PDP) based on the mathematical
formulation by Apley and Zhu (2020) and Christoph Molnar (2022).

All implementations are fully vectorized using NumPy for high computational
efficiency, strict mean-zero centering against the empirical sample distribution,
and robust two-way ANOVA centering with observation masking for 2D spatial interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ALE1DResult:
    """Container for 1D Accumulated Local Effects computation output."""

    feature_name: str
    grid_values: np.ndarray
    ale_values: np.ndarray
    sample_values: np.ndarray
    bin_counts: np.ndarray

    @property
    def effect_range(self) -> float:
        """Return the peak-to-peak range of the ALE curve."""
        return float(np.ptp(self.ale_values))


@dataclass(frozen=True)
class ALE2DResult:
    """Container for 2D Accumulated Local Effects computation output."""

    feature_x: str
    feature_y: str
    grid_x: np.ndarray
    grid_y: np.ndarray
    ale_grid: np.ndarray
    cell_counts: np.ndarray


@dataclass(frozen=True)
class PDP1DResult:
    """Container for 1D Partial Dependence computation output."""

    feature_name: str
    grid_values: np.ndarray
    pdp_raw: np.ndarray
    pdp_centered: np.ndarray

    @property
    def effect_range(self) -> float:
        """Return the peak-to-peak range of the centered PDP curve."""
        return float(np.ptp(self.pdp_centered))


class ALEEngine:
    """Engine for computing 1D and 2D Accumulated Local Effects and Partial Dependence."""

    @staticmethod
    def compute_1d(
        predict_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray],
        data: pd.DataFrame,
        feature: str,
        grid_size: int = 40,
    ) -> ALE1DResult:
        """Compute first-order centered Accumulated Local Effects (1D ALE) using vectorization.

        Parameters
        ----------
        predict_fn : Callable
            Prediction function of the trained regression model.
        data : pd.DataFrame
            Evaluation data matrix (e.g., test set).
        feature : str
            Column name of the feature to analyze.
        grid_size : int, default=40
            Target number of quantile bins.

        Returns
        -------
        ALE1DResult
            Data class with grid points, centered ALE values, and sample data.
        """
        x_raw = data[feature].to_numpy(dtype=float)
        quantiles = np.linspace(0.0, 1.0, grid_size + 1)
        grid = np.unique(np.quantile(x_raw, quantiles))

        num_bins = len(grid) - 1
        if num_bins < 2:
            raise ValueError(
                f"Feature '{feature}' does not have enough unique values for ALE binning."
            )

        # Assign observations into intervals [grid[k], grid[k+1]) (0-indexed: 0 to num_bins-1)
        bin_indices = np.digitize(x_raw, grid[1:-1])

        # Vectorized batch evaluation across all samples at bin lower and upper boundaries
        subset_low = data.copy()
        subset_high = data.copy()
        subset_low[feature] = grid[bin_indices]
        subset_high[feature] = grid[bin_indices + 1]

        pred_low = predict_fn(subset_low)
        pred_high = predict_fn(subset_high)
        diff = pred_high - pred_low

        bin_counts = np.bincount(bin_indices, minlength=num_bins)
        sum_diffs = np.bincount(bin_indices, weights=diff, minlength=num_bins)

        uncentered_diffs = np.zeros(num_bins, dtype=float)
        valid = bin_counts > 0
        uncentered_diffs[valid] = sum_diffs[valid] / bin_counts[valid]

        # Accumulate local differences across bins
        accumulated = np.zeros(num_bins + 1, dtype=float)
        accumulated[1:] = np.cumsum(uncentered_diffs)

        # Center the ALE curve using linear interpolation over empirical sample distribution
        # to ensure E[ALE(X)] = 0
        interpolated = np.interp(x_raw, grid, accumulated)
        centering_constant = float(np.mean(interpolated))
        centered_ale = accumulated - centering_constant

        return ALE1DResult(
            feature_name=feature,
            grid_values=grid,
            ale_values=centered_ale,
            sample_values=x_raw,
            bin_counts=bin_counts,
        )

    @staticmethod
    def compute_2d(
        predict_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray],
        data: pd.DataFrame,
        feature_x: str,
        feature_y: str,
        grid_size: int = 20,
        min_samples: int = 5,
    ) -> ALE2DResult:
        """Compute second-order centered Accumulated Local Effects (2D ALE).

        Implements vectorized second differences, dual cumsum accumulation,
        and two-way weighted additive centering (Apley & Zhu, 2020) so that marginal
        first-order effects are completely removed, isolating pure interaction.
        Cells with fewer than min_samples observations are masked with NaN.

        Parameters
        ----------
        predict_fn : Callable
            Prediction function of the trained model.
        data : pd.DataFrame
            Evaluation dataset.
        feature_x : str
            First feature name (X-axis).
        feature_y : str
            Second feature name (Y-axis).
        grid_size : int, default=20
            Number of quantile bins per axis.
        min_samples : int, default=5
            Minimum observations required in a cell to be considered valid and plotted.

        Returns
        -------
        ALE2DResult
            Data class containing 2D grid coordinates, centered interaction effects,
            and cell observation counts.
        """
        x1_raw = data[feature_x].to_numpy(dtype=float)
        x2_raw = data[feature_y].to_numpy(dtype=float)

        grid_x = np.unique(np.quantile(x1_raw, np.linspace(0.0, 1.0, grid_size + 1)))
        grid_y = np.unique(np.quantile(x2_raw, np.linspace(0.0, 1.0, grid_size + 1)))

        num_x = len(grid_x) - 1
        num_y = len(grid_y) - 1

        bin_x = np.digitize(x1_raw, grid_x[1:-1])
        bin_y = np.digitize(x2_raw, grid_y[1:-1])

        # Vectorized evaluation at the 4 cell vertices
        c11 = data.copy()
        c11[feature_x] = grid_x[bin_x + 1]
        c11[feature_y] = grid_y[bin_y + 1]

        c10 = data.copy()
        c10[feature_x] = grid_x[bin_x + 1]
        c10[feature_y] = grid_y[bin_y]

        c01 = data.copy()
        c01[feature_x] = grid_x[bin_x]
        c01[feature_y] = grid_y[bin_y + 1]

        c00 = data.copy()
        c00[feature_x] = grid_x[bin_x]
        c00[feature_y] = grid_y[bin_y]

        p11 = predict_fn(c11)
        p10 = predict_fn(c10)
        p01 = predict_fn(c01)
        p00 = predict_fn(c00)

        second_diff = (p11 - p10) - (p01 - p00)

        flat_bins = bin_x * num_y + bin_y
        counts_flat = np.bincount(flat_bins, minlength=num_x * num_y)
        sum_diff_flat = np.bincount(flat_bins, weights=second_diff, minlength=num_x * num_y)

        delta = np.zeros(num_x * num_y, dtype=float)
        valid = counts_flat > 0
        delta[valid] = sum_diff_flat[valid] / counts_flat[valid]
        delta = delta.reshape((num_x, num_y))
        cell_counts = counts_flat.reshape((num_x, num_y))

        # 2D cumulative sum over grid cells
        accumulated = np.zeros((num_x + 1, num_y + 1), dtype=float)
        accumulated[1:, 1:] = np.cumsum(np.cumsum(delta, axis=0), axis=1)

        # Cell-level accumulated effects
        cell_accum = accumulated[1:, 1:].copy()

        # Two-way weighted additive centering (WLS ANOVA decomposition via alternating projections)
        # to guarantee zero marginals: E[f_{1,2}(X1, x2)|X2=x2] = 0 and E[f_{1,2}(x1, X2)|X1=x1] = 0
        w = cell_counts.astype(float)
        mask_obs = w > 0

        alpha = np.zeros(num_x)
        beta = np.zeros(num_y)

        for _ in range(100):
            for i in range(num_x):
                row_mask = mask_obs[i, :]
                row_sum = np.sum(w[i, row_mask])
                if row_sum > 0:
                    alpha[i] = np.sum(w[i, row_mask] * (cell_accum[i, row_mask] - beta[row_mask])) / row_sum
            for j in range(num_y):
                col_mask = mask_obs[:, j]
                col_sum = np.sum(w[col_mask, j])
                if col_sum > 0:
                    beta[j] = np.sum(w[col_mask, j] * (cell_accum[col_mask, j] - alpha[col_mask])) / col_sum

        # Compute pure second-order interaction on cells
        # Mask sparse cells (< min_samples) with NaN
        interaction_cells = np.full((num_x, num_y), np.nan, dtype=float)
        for i in range(num_x):
            for j in range(num_y):
                if cell_counts[i, j] >= min_samples:
                    interaction_cells[i, j] = cell_accum[i, j] - alpha[i] - beta[j]

        return ALE2DResult(
            feature_x=feature_x,
            feature_y=feature_y,
            grid_x=grid_x,
            grid_y=grid_y,
            ale_grid=interaction_cells,
            cell_counts=cell_counts,
        )

    @staticmethod
    def compute_pdp_1d(
        predict_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray],
        data: pd.DataFrame,
        feature: str,
        grid_size: int = 40,
        subsample: int = 1000,
        grid: np.ndarray | None = None,
    ) -> PDP1DResult:
        """Compute 1D Partial Dependence Plot (PDP) with distribution-weighted centering.

        Parameters
        ----------
        predict_fn : Callable
            Prediction function of the trained regression model.
        data : pd.DataFrame
            Evaluation data matrix.
        feature : str
            Feature column name.
        grid_size : int, default=40
            Number of evaluation points if grid is not provided.
        subsample : int, default=1000
            Subsample size for computational efficiency.
        grid : np.ndarray | None, default=None
            Custom grid points (e.g. matching quantile grid from ALE) for fair comparison.

        Returns
        -------
        PDP1DResult
            Data class with PDP evaluation points, raw, and centered values.
        """
        x_raw = data[feature].to_numpy(dtype=float)
        if grid is None:
            grid = np.linspace(np.min(x_raw), np.max(x_raw), grid_size)
        else:
            grid = np.asarray(grid, dtype=float)

        eval_data = data
        if len(data) > subsample:
            eval_data = data.sample(n=subsample, random_state=42)

        pdp_values = np.zeros(len(grid), dtype=float)
        for idx, val in enumerate(grid):
            synthetic_df = eval_data.copy()
            synthetic_df[feature] = val
            preds = predict_fn(synthetic_df)
            pdp_values[idx] = float(np.mean(preds))

        # Consistent centering: center by expectation over empirical sample distribution
        # to ensure fair comparison with ALE centering E[PDP(X)] = 0
        sample_interp = np.interp(x_raw, grid, pdp_values)
        centering_constant = float(np.mean(sample_interp))
        centered_pdp = pdp_values - centering_constant

        return PDP1DResult(
            feature_name=feature,
            grid_values=grid,
            pdp_raw=pdp_values,
            pdp_centered=centered_pdp,
        )
