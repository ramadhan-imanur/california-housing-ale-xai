"""Training, Evaluation, and Accumulated Local Effects (ALE) Analysis Pipeline.

This script executes the complete experimental workflow for California Housing
Prices: data preprocessing, baseline model training (Random Forest and Gradient
Boosting), evaluation metrics, 1D and 2D ALE computation, and comparison with
Partial Dependence Plots (PDP).
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ensure src/ is accessible in Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CODE_DIR = SCRIPT_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ale_engine import ALEEngine


def load_and_preprocess_data(
    data_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load raw housing CSV and perform imputation, encoding, and train-test split."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    raw_df = pd.read_csv(data_path)

    # Impute missing values in total_bedrooms using median
    median_bedrooms = float(raw_df["total_bedrooms"].median())
    processed_df = raw_df.copy()
    processed_df["total_bedrooms"] = processed_df["total_bedrooms"].fillna(
        median_bedrooms
    )

    # One-hot encode ocean_proximity
    processed_df = pd.get_dummies(
        processed_df, columns=["ocean_proximity"], drop_first=True, dtype=float
    )

    X = processed_df.drop(columns=["median_house_value"])
    y = processed_df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    return X_train, y_train, X_test, y_test, raw_df


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[RandomForestRegressor, HistGradientBoostingRegressor]:
    """Train Random Forest and Gradient Boosting regressors."""
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    gb_model = HistGradientBoostingRegressor(max_iter=150, random_state=42)
    gb_model.fit(X_train, y_train)

    return rf_model, gb_model


def evaluate_model(
    model: RandomForestRegressor | HistGradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calculate R2, RMSE, and MAE for a trained model."""
    predictions = model.predict(X_test)
    r2 = float(r2_score(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def generate_visualizations(
    raw_df: pd.DataFrame,
    best_model: HistGradientBoostingRegressor,
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

        sample_subset = np.random.choice(
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

    # 3. Comparative Analysis: ALE vs PDP for median_income
    ale_income = ALEEngine.compute_1d(
        best_model.predict, X_test, "median_income", grid_size=40
    )
    pdp_income = ALEEngine.compute_pdp_1d(
        best_model.predict, X_test, "median_income", grid_size=40
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        ale_income.grid_values,
        ale_income.ale_values,
        label="Accumulated Local Effects (ALE)",
        color="#1f77b4",
        lw=2.5,
    )
    plt.plot(
        pdp_income.grid_values,
        pdp_income.pdp_centered,
        label="Partial Dependence Plot (Centered PDP)",
        color="#d62728",
        linestyle="--",
        lw=2.5,
    )
    plt.axhline(0, color="gray", linestyle=":", alpha=0.8)
    plt.title(
        "Comparison of ALE vs. PDP on median_income (Overcoming Correlated Feature Bias)",
        fontsize=12,
        pad=10,
    )
    plt.xlabel("Median Income ($10,000 USD)")
    plt.ylabel("Centered Main Effect on Median House Value (USD)")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "04_ale_vs_pdp_median_income.png", dpi=300)
    plt.close()

    # 4. Comparative Analysis: ALE vs PDP for longitude
    ale_long = ALEEngine.compute_1d(
        best_model.predict, X_test, "longitude", grid_size=40
    )
    pdp_long = ALEEngine.compute_pdp_1d(
        best_model.predict, X_test, "longitude", grid_size=40
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        ale_long.grid_values,
        ale_long.ale_values,
        label="Accumulated Local Effects (ALE)",
        color="#1f77b4",
        lw=2.5,
    )
    plt.plot(
        pdp_long.grid_values,
        pdp_long.pdp_centered,
        label="Partial Dependence Plot (Centered PDP)",
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

    # 5. 2D Spatial Interaction ALE (longitude x latitude)
    ale_2d = ALEEngine.compute_2d(
        best_model.predict, X_test, "longitude", "latitude", grid_size=20
    )
    plt.figure(figsize=(10, 7.5))
    X_mesh, Y_mesh = np.meshgrid(ale_2d.grid_x, ale_2d.grid_y)
    contour = plt.contourf(
        X_mesh,
        Y_mesh,
        ale_2d.ale_grid.T,
        levels=25,
        cmap="viridis",
    )
    plt.colorbar(contour, label="Second-Order ALE Interaction Effect (USD)")

    sample_geo = X_test.sample(min(800, len(X_test)), random_state=42)
    plt.scatter(
        sample_geo["longitude"],
        sample_geo["latitude"],
        color="red",
        s=4,
        alpha=0.3,
        label="Observed Census Blocks",
    )

    plt.title(
        "2D Spatial ALE Interaction: Longitude × Latitude (True Geographic Landmass)",
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
    candidate_paths = [
        CODE_DIR / "data" / "california_housing_prices.csv",
        PROJECT_ROOT / "data" / "california_housing_prices.csv",
        PROJECT_ROOT / ".raw" / "california_housing_prices.csv",
    ]
    data_path = next((p for p in candidate_paths if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Dataset california_housing_prices.csv tidak ditemukan di "
            f"code/data, {PROJECT_ROOT}/data, maupun {PROJECT_ROOT}/.raw!"
        )
    outputs_dir = CODE_DIR / "outputs"
    plots_dir = outputs_dir / "plots"

    print("Step 1: Loading and preprocessing California Housing data...")
    X_train, y_train, X_test, y_test, raw_df = load_and_preprocess_data(data_path)
    print(
        f"Data ready: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples."
    )

    print("\nStep 2: Training regression baseline models...")
    rf_model, gb_model = train_models(X_train, y_train)

    print("\nStep 3: Evaluating performance metrics on unseen test set...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    gb_metrics = evaluate_model(gb_model, X_test, y_test)

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
    generate_visualizations(raw_df, gb_model, X_test, plots_dir)
    print(f"All visualizations successfully saved to: {plots_dir}")


if __name__ == "__main__":
    main()
