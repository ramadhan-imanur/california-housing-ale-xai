"""Accumulated Local Effects (ALE) and Partial Dependence Engine.

This module implements first-order (1D) and second-order (2D) Accumulated Local
Effects (ALE) alongside Partial Dependence Plots (PDP) based on the mathematical
formulation by Apley and Zhu (2020) and Christoph Molnar (2022).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

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


class ALEEngine:
    """Engine for computing 1D and 2D Accumulated Local Effects."""

    @staticmethod
    def compute_1d(
        predict_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray],
        data: pd.DataFrame,
        feature: str,
        grid_size: int = 40,
    ) -> ALE1DResult:
        """Compute first-order centered Accumulated Local Effects (1D ALE).

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

        # Assign observations into intervals (1-indexed bins: 1 to num_bins)
        bin_indices = np.digitize(x_raw, grid[1:-1])

        uncentered_diffs = np.zeros(num_bins, dtype=float)
        bin_counts = np.zeros(num_bins, dtype=int)

        # Batch evaluation for each interval
        for k in range(num_bins):
            mask = bin_indices == k
            count = int(np.sum(mask))
            bin_counts[k] = count
            if count == 0:
                continue

            subset = data.loc[mask].copy()

            subset_low = subset.copy()
            subset_low[feature] = grid[k]

            subset_high = subset.copy()
            subset_high[feature] = grid[k + 1]

            pred_high = predict_fn(subset_high)
            pred_low = predict_fn(subset_low)

            uncentered_diffs[k] = float(np.mean(pred_high - pred_low))

        # Accumulate local differences across bins
        accumulated = np.zeros(num_bins + 1, dtype=float)
        accumulated[1:] = np.cumsum(uncentered_diffs)

        # Center the ALE curve using linear interpolation over sample values
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
    ) -> ALE2DResult:
        """Compute second-order centered Accumulated Local Effects (2D ALE).

        Parameters
        ----------
        predict_fn : Callable
            Prediction function of the trained model.
        data : pd.DataFrame
            Evaluation dataset.
        feature_x : str
            First feature name.
        feature_y : str
            Second feature name.
        grid_size : int, default=20
            Number of quantile bins per axis.

        Returns
        -------
        ALE2DResult
            Data class containing 2D grid coordinates and centered interaction effects.
        """
        x1_raw = data[feature_x].to_numpy(dtype=float)
        x2_raw = data[feature_y].to_numpy(dtype=float)

        grid_x = np.unique(np.quantile(x1_raw, np.linspace(0.0, 1.0, grid_size + 1)))
        grid_y = np.unique(np.quantile(x2_raw, np.linspace(0.0, 1.0, grid_size + 1)))

        num_x = len(grid_x) - 1
        num_y = len(grid_y) - 1

        bin_x = np.digitize(x1_raw, grid_x[1:-1])
        bin_y = np.digitize(x2_raw, grid_y[1:-1])

        delta = np.zeros((num_x, num_y), dtype=float)
        cell_counts = np.zeros((num_x, num_y), dtype=int)

        for i in range(num_x):
            for j in range(num_y):
                mask = (bin_x == i) & (bin_y == j)
                count = int(np.sum(mask))
                cell_counts[i, j] = count
                if count == 0:
                    continue

                subset = data.loc[mask]

                # 4 corners of the 2D cell
                c11 = subset.copy()
                c11[feature_x] = grid_x[i + 1]
                c11[feature_y] = grid_y[j + 1]

                c10 = subset.copy()
                c10[feature_x] = grid_x[i + 1]
                c10[feature_y] = grid_y[j]

                c01 = subset.copy()
                c01[feature_x] = grid_x[i]
                c01[feature_y] = grid_y[j + 1]

                c00 = subset.copy()
                c00[feature_x] = grid_x[i]
                c00[feature_y] = grid_y[j]

                p11 = predict_fn(c11)
                p10 = predict_fn(c10)
                p01 = predict_fn(c01)
                p00 = predict_fn(c00)

                second_diff = (p11 - p10) - (p01 - p00)
                delta[i, j] = float(np.mean(second_diff))

        # Accumulate over 2D grid
        accumulated = np.zeros((num_x + 1, num_y + 1), dtype=float)
        for i in range(num_x):
            for j in range(num_y):
                accumulated[i + 1, j + 1] = (
                    accumulated[i, j + 1]
                    + accumulated[i + 1, j]
                    - accumulated[i, j]
                    + delta[i, j]
                )

        # Center the 2D surface
        centering_constant = float(np.mean(accumulated[:-1, :-1][cell_counts > 0]))
        centered_grid = accumulated - centering_constant

        return ALE2DResult(
            feature_x=feature_x,
            feature_y=feature_y,
            grid_x=grid_x,
            grid_y=grid_y,
            ale_grid=centered_grid,
            cell_counts=cell_counts,
        )

    @staticmethod
    def compute_pdp_1d(
        predict_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray],
        data: pd.DataFrame,
        feature: str,
        grid_size: int = 40,
        subsample: int = 1000,
    ) -> PDP1DResult:
        """Compute standard 1D Partial Dependence Plot (PDP).

        Parameters
        ----------
        predict_fn : Callable
            Prediction function.
        data : pd.DataFrame
            Evaluation data matrix.
        feature : str
            Feature column name.
        grid_size : int, default=40
            Number of evaluation points.
        subsample : int, default=1000
            Subsample size for computational efficiency.

        Returns
        -------
        PDP1DResult
            Data class with PDP evaluation points, raw, and centered values.
        """
        x_raw = data[feature].to_numpy(dtype=float)
        grid = np.linspace(np.min(x_raw), np.max(x_raw), grid_size)

        eval_data = data
        if len(data) > subsample:
            eval_data = data.sample(n=subsample, random_state=42)

        pdp_values = np.zeros(grid_size, dtype=float)
        for idx, val in enumerate(grid):
            synthetic_df = eval_data.copy()
            synthetic_df[feature] = val
            preds = predict_fn(synthetic_df)
            pdp_values[idx] = float(np.mean(preds))

        centered_pdp = pdp_values - float(np.mean(pdp_values))

        return PDP1DResult(
            feature_name=feature,
            grid_values=grid,
            pdp_raw=pdp_values,
            pdp_centered=centered_pdp,
        )
