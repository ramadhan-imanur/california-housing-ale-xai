"""Training, Evaluation, and Accumulated Local Effects (ALE) Analysis Pipeline.

This script executes the complete experimental workflow for California Housing
Prices: data preprocessing, pipeline construction with domain ratio feature engineering,
baseline model training (Random Forest and HistGradientBoosting with early stopping),
evaluation metrics calculation, and high-resolution publication-ready visualizations:
- 01: Correlation Matrix Heatmap
- 03: 1D ALE Across Continuous Features (with seeded reproducible rug plots)
- 04: Fair ALE vs. Centered PDP Comparison on median_income
- 05: ALE vs. Centered PDP Comparison on longitude (spatial collinearity)
- 06: Pure 2D Spatial Interaction ALE (Longitude x Latitude) with two-way centering & cell masking
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Ensure src/ is accessible in Python path
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ale_engine import ALEEngine


class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain ratio feature transformer evaluated inside scikit-learn pipeline."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HousingFeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out["rooms_per_household"] = X_out["total_rooms"] / (X_out["households"] + 1e-6)
        X_out["bedrooms_per_room"] = X_out["total_bedrooms"] / (X_out["total_rooms"] + 1e-6)
        X_out["population_per_household"] = X_out["population"] / (X_out["households"] + 1e-6)
        return X_out


def load_and_preprocess_data(
    data_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load raw housing CSV and perform encoding and clean train-test split without pre-split leakage."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    raw_df = pd.read_csv(data_path)

    # One-hot encode ocean_proximity categorical column
    processed_df = pd.get_dummies(
        raw_df, columns=["ocean_proximity"], drop_first=True, dtype=float
    )

    X = processed_df.drop(columns=["median_house_value"])
    y = processed_df["median_house_value"]

    # Partition dataset into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    return X_train, y_train, X_test, y_test, raw_df


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[Pipeline, Pipeline]:
    """Train Random Forest and Gradient Boosting regressors encapsulated in pipelines."""
    rf_pipeline = Pipeline([
        ("feature_engineer", HousingFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
            ),
        ),
    ])
    rf_pipeline.fit(X_train, y_train)

    gb_pipeline = Pipeline([
        ("feature_engineer", HousingFeatureEngineer()),
        (
            "regressor",
            HistGradientBoostingRegressor(
                max_iter=500, early_stopping=True, random_state=42
            ),
        ),
    ])
    gb_pipeline.fit(X_train, y_train)

    return rf_pipeline, gb_pipeline


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calculate R2, RMSE, and MAE for a trained model pipeline."""
    predictions = model.predict(X_test)
    r2 = float(r2_score(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def generate_visualizations(
    raw_df: pd.DataFrame,
    best_model: Pipeline,
    X_test: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Generate and save publication-ready figures."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    # 1. Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
    corr_matrix = raw_df[numeric_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    plt.title("Correlation Matrix of California Housing Attributes", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "01_correlation_matrix.png", dpi=300)
    plt.close()

    # 2. 1D ALE Curves across major features
    features_to_plot = [
        "median_income",
        "housing_median_age",
        "total_rooms",
        "population",
        "longitude",
        "latitude",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    rng = np.random.default_rng(42)

    for idx, feat in enumerate(features_to_plot):
        ale_res = ALEEngine.compute_1d(best_model.predict, X_test, feat, grid_size=40)
        ax = axes[idx]
        ax.plot(
            ale_res.grid_values,
            ale_res.ale_values,
            color="#1f77b4",
            lw=2.5,
            label=f"ALE (range={ale_res.effect_range:,.0f})",
        )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)

        sample_subset = rng.choice(
            ale_res.sample_values,
            size=min(250, len(ale_res.sample_values)),
            replace=False,
        )
        ax.plot(
            sample_subset,
            np.full_like(sample_subset, np.min(ale_res.ale_values)),
            "|",
            color="black",
            alpha=0.5,
            markersize=8,
        )

        ax.set_title(f"1D ALE: {feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Centered ALE Effect (USD)")
        ax.legend(fontsize=9)

    plt.suptitle(
        "Accumulated Local Effects (ALE) Across Six Continuous Predictors",
        fontsize=14,
        y=0.99,
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "03_ale_1d_all_features.png", dpi=300)
    plt.close()

    # 3. Comparative Analysis: ALE vs PDP for median_income (fair quantile grid & consistent centering)
    ale_income = ALEEngine.compute_1d(
        best_model.predict, X_test, "median_income", grid_size=40
    )
    pdp_income = ALEEngine.compute_pdp_1d(
        best_model.predict,
        X_test,
        "median_income",
        grid=ale_income.grid_values,
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        ale_income.grid_values,
        ale_income.ale_values,
        label=f"Accumulated Local Effects (range=${ale_income.effect_range:,.0f})",
        color="#1f77b4",
        lw=2.5,
    )
    plt.plot(
        pdp_income.grid_values,
        pdp_income.pdp_centered,
        label=f"Partial Dependence Plot (Centered PDP, range=${pdp_income.effect_range:,.0f})",
        color="#d62728",
        linestyle="--",
        lw=2.5,
    )
    plt.axhline(0, color="gray", linestyle=":", alpha=0.8)
    plt.title(
        "Comparison of ALE vs. Centered PDP on median_income (Consistent Grid & Centering)",
        fontsize=12,
        pad=10,
    )
    plt.xlabel("Median Income ($10,000 USD)")
    plt.ylabel("Centered Main Effect on Median House Value (USD)")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "04_ale_vs_pdp_median_income.png", dpi=300)
    plt.close()

    # 4. Comparative Analysis: ALE vs PDP for longitude (Spatial Coordinate Collinearity)
    ale_long = ALEEngine.compute_1d(
        best_model.predict, X_test, "longitude", grid_size=40
    )
    pdp_long = ALEEngine.compute_pdp_1d(
        best_model.predict,
        X_test,
        "longitude",
        grid=ale_long.grid_values,
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        ale_long.grid_values,
        ale_long.ale_values,
        label=f"Accumulated Local Effects (range=${ale_long.effect_range:,.0f})",
        color="#1f77b4",
        lw=2.5,
    )
    plt.plot(
        pdp_long.grid_values,
        pdp_long.pdp_centered,
        label=f"Partial Dependence Plot (Centered PDP, range=${pdp_long.effect_range:,.0f})",
        color="#d62728",
        linestyle="--",
        lw=2.5,
    )
    plt.axhline(0, color="gray", linestyle=":", alpha=0.8)
    plt.title(
        "Comparison of ALE vs. PDP on longitude (Spatial Coordinate Collinearity)",
        fontsize=12,
        pad=10,
    )
    plt.xlabel("Longitude (Degrees)")
    plt.ylabel("Centered Effect on Median House Value (USD)")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "05_ale_vs_pdp_longitude.png", dpi=300)
    plt.close()

    # 5. Pure 2D Spatial Interaction ALE (longitude x latitude) with cell masking & pcolormesh
    ale_2d = ALEEngine.compute_2d(
        best_model.predict, X_test, "longitude", "latitude", grid_size=20, min_samples=5
    )
    plt.figure(figsize=(10, 7.5))

    # Masked array for pcolormesh
    masked_grid = np.ma.masked_invalid(ale_2d.ale_grid.T)
    mesh = plt.pcolormesh(
        ale_2d.grid_x,
        ale_2d.grid_y,
        masked_grid,
        cmap="viridis",
        shading="flat",
    )
    plt.colorbar(mesh, label="Pure Second-Order ALE Interaction Effect (USD)")

    sample_geo = X_test.sample(min(800, len(X_test)), random_state=42)
    plt.scatter(
        sample_geo["longitude"],
        sample_geo["latitude"],
        color="red",
        s=5,
        alpha=0.4,
        label="Observed Census Blocks",
    )

    valid_cells_count = int(np.sum(~np.isnan(ale_2d.ale_grid)))
    total_cells_count = ale_2d.cell_counts.size

    plt.title(
        f"Pure 2D Spatial ALE Interaction: Longitude × Latitude ({valid_cells_count}/{total_cells_count} Cells ≥ 5 Obs)",
        fontsize=12,
        pad=10,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(plots_dir / "06_ale_2d_spatial_interaction.png", dpi=300)
    plt.close()


def main() -> None:
    """Execute end-to-end training, evaluation, and ALE export."""
    data_path = CODE_DIR / "data" / "california_housing_prices.csv"
    if not data_path.exists():
        fallback = Path(__file__).resolve().parents[2] / "data" / "california_housing_prices.csv"
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

    outputs_dir = CODE_DIR / "outputs"
    plots_dir = outputs_dir / "plots"

    print("Step 1: Loading and preprocessing California Housing data...")
    X_train, y_train, X_test, y_test, raw_df = load_and_preprocess_data(data_path)
    print(
        f"Data ready: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples."
    )

    print("\nStep 2: Training regression baseline pipelines with domain ratio features...")
    rf_pipeline, gb_pipeline = train_models(X_train, y_train)

    hgb_step = gb_pipeline.named_steps["regressor"]
    print(f"HistGradientBoosting converged after {hgb_step.n_iter_} iterations.")

    print("\nStep 3: Evaluating performance metrics on unseen test set...")
    rf_metrics = evaluate_model(rf_pipeline, X_test, y_test)
    gb_metrics = evaluate_model(gb_pipeline, X_test, y_test)

    summary = {
        "RandomForestRegressor": rf_metrics,
        "HistGradientBoostingRegressor": gb_metrics,
    }

    print("-" * 55)
    print(f"{'Model':<30} | {'R2':<8} | {'RMSE (USD)':<10} | {'MAE (USD)':<10}")
    print("-" * 55)
    for model_name, metrics in summary.items():
        print(
            f"{model_name:<30} | {metrics['R2']:<8.4f} | {metrics['RMSE']:<10.2f} | {metrics['MAE']:<10.2f}"
        )
    print("-" * 55)

    metrics_path = outputs_dir / "results_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("\nStep 4: Computing ALE curves and generating publication-ready figures...")
    generate_visualizations(raw_df, gb_pipeline, X_test, plots_dir)
    print(f"All visualizations successfully saved to: {plots_dir}")


if __name__ == "__main__":
    main()
