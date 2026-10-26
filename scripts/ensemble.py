import numpy as np
import pandas as pd
from scripts.models import Mbj, forecast_metrics


def build_rolling_stacking(
    diagnostics_predictions,
    feature_set="prices_wti_fundamentals",
    train_window=180,
    base_models=("ridge_cv", "random_forest", "lightgbm"),
    stack_window=60,
    min_periods=24,
):
    """
    Constructs walk-forward ensemble predictions using learned combination weights.

    Parameters
    ----------
    diagnostics_predictions : dict
        Output from run_overfitting_diagnostics.
    feature_set : str
        Information set to ensemble across models.
    train_window : int
        Base model training window.
    base_models : tuple
        Names of the models to combine.
    stack_window : int, default=60
        Rolling memory (months) for learning ensemble weights.
    min_periods : int, default=24
        Warm-up periods before learning weights (uses 1/N equal weight during warm-up).

    Returns
    -------
    pd.DataFrame
        Ensemble forecasts (Equal Weight, Stacking Mbj Unconstrained, Stacking Mbj Positive).
    pd.DataFrame
        Historical weights assigned to each base model over time.
    """
    # 1. Align Out-of-Sample Predictions across models
    preds_dfs = []
    for m in base_models:
        key = f"{feature_set}__{m}__window_{train_window}"
        if key not in diagnostics_predictions:
            raise KeyError(f"Forecast key '{key}' not found in predictions dictionary.")
        df_m = diagnostics_predictions[key]["oos"][["date", "y_true", "y_pred"]].copy()
        df_m = df_m.rename(columns={"y_pred": f"pred_{m}"})
        preds_dfs.append(df_m.set_index("date"))

    combined = pd.concat(preds_dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()].dropna().reset_index()

    pred_cols = [f"pred_{m}" for m in base_models]
    n_models = len(base_models)
    dates = combined["date"].values
    y_true = combined["y_true"].values
    preds_matrix = combined[pred_cols].values

    # Storage arrays
    y_pred_equal = []
    y_pred_mbj_unconstrained = []
    y_pred_mbj_positive = []

    weights_unconstrained = []
    weights_positive = []

    # 2. Walk-Forward Weight Learning Loop
    for t in range(len(combined)):
        # Equal Weight Baseline: 1 / N
        y_pred_equal.append(np.mean(preds_matrix[t, :]))

        # Warm-up phase: default to equal weights
        if t < min_periods:
            y_pred_mbj_unconstrained.append(np.mean(preds_matrix[t, :]))
            y_pred_mbj_positive.append(np.mean(preds_matrix[t, :]))
            weights_unconstrained.append(np.full(n_models, 1.0 / n_models))
            weights_positive.append(np.full(n_models, 1.0 / n_models))
            continue

        # Lookback memory window for stacking
        start_idx = max(0, t - stack_window)
        X_stack = preds_matrix[start_idx:t, :]
        y_stack = y_true[start_idx:t]

        # A. Unconstrained Britten-Jones Estimator
        mbj_uncons = Mbj(positive=False, normalize=True).fit(X_stack, y_stack)
        w_uncons = mbj_uncons.coef_
        pred_uncons = np.dot(preds_matrix[t, :], w_uncons)
        y_pred_mbj_unconstrained.append(pred_uncons)
        weights_unconstrained.append(w_uncons)

        # B. Positive-Constrained Britten-Jones Estimator (No negative weights)
        mbj_pos = Mbj(positive=True, normalize=True).fit(X_stack, y_stack)
        w_pos = mbj_pos.coef_
        pred_pos = np.dot(preds_matrix[t, :], w_pos)
        y_pred_mbj_positive.append(pred_pos)
        weights_positive.append(w_pos)

    # 3. Format Output DataFrames
    ensemble_forecasts = pd.DataFrame({
        "date": dates,
        "y_true": y_true,
        "pred_equal_weight": y_pred_equal,
        "pred_mbj_unconstrained": y_pred_mbj_unconstrained,
        "pred_mbj_positive": y_pred_mbj_positive,
    })

    weights_df = pd.DataFrame(
        weights_positive,
        index=dates,
        columns=[f"weight_{m}" for m in base_models]
    ).reset_index().rename(columns={"index": "date"})

    return ensemble_forecasts, weights_df


def add_ensembles_to_diagnostics(diagnostics_df, diagnostics_predictions, feature_sets, train_windows):
    """
    Calcule automatiquement les 3 modèles d'Ensemble (1/N, Mbj Unconstrained, Mbj Positive)
    pour toutes les fenêtres et jeux de données, et les fusionne dans diagnostics_df et diagnostics_predictions.
    """
    new_records = []
    
    for window in train_windows:
        for fset in feature_sets.keys():
            ensemble_forecasts, _ = build_rolling_stacking(
                diagnostics_predictions=diagnostics_predictions,
                feature_set=fset,
                train_window=window,
                base_models=("ridge_cv", "random_forest", "lightgbm"),
                stack_window=60,
                min_periods=24,
            )
            
            strats = {
                "ensemble_1_over_n": "pred_equal_weight",
                "ensemble_mbj_unconstrained": "pred_mbj_unconstrained",
                "ensemble_mbj_positive": "pred_mbj_positive",
            }
            
            for model_name, col in strats.items():
                oos_df = pd.DataFrame({
                    "date": pd.to_datetime(ensemble_forecasts["date"]),
                    "y_true": ensemble_forecasts["y_true"],
                    "y_pred": ensemble_forecasts[col]
                }).dropna().sort_values("date").reset_index(drop=True)
                
                key = f"{fset}__{model_name}__window_{window}"
                diagnostics_predictions[key] = {"oos": oos_df, "train": None}
                
                m = forecast_metrics(oos_df)
                
                new_records.append({
                    "train_window": window,
                    "feature_set": fset,
                    "model": model_name,
                    "n_features": len(feature_sets[fset]),
                    "n_oos": len(oos_df),
                    "train_rmse": np.nan,
                    "oos_rmse": m["rmse"],
                    "rmse_gap": np.nan,
                    "train_mae": np.nan,
                    "oos_mae": m["mae"],
                    "train_r2": np.nan,
                    "oos_r2": m["r2"],
                    "r2_gap": np.nan,
                    "train_correlation": np.nan,
                    "oos_correlation": m["correlation"],
                    "train_directional_accuracy": np.nan,
                    "oos_directional_accuracy": m["directional_accuracy"],
                    "oos_start": oos_df["date"].min(),
                    "oos_end": oos_df["date"].max(),
                })

    full_diagnostics_df = pd.concat([diagnostics_df, pd.DataFrame(new_records)], ignore_index=True)
    return full_diagnostics_df, diagnostics_predictions