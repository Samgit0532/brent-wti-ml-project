import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def rolling_benchmarks(
    data,
    evaluation_start,
    train_window=120,
    target_col="target_spread_change",
):
    """Generate one-month-ahead benchmark forecasts."""
    df = data.sort_values("date").reset_index(drop=True).copy()
    evaluation_start = pd.Timestamp(evaluation_start)

    forecasts = []

    for i in df.index[df["date"] >= evaluation_start]:
        if i < train_window:
            raise ValueError(
                f"Insufficient training history at {df.loc[i, 'date']}."
            )

        # The preceding row's target is known at the current decision date
        train = df.iloc[i - train_window:i]
        current = df.iloc[[i]]

        ar1 = LinearRegression()
        ar1.fit(
            train[["spread_change"]],
            train[target_col],
        )

        predictions = {
            "zero_change": 0.0,
            "historical_mean": train[target_col].mean(),
            "ar1": ar1.predict(current[["spread_change"]])[0],
        }

        for name, prediction in predictions.items():
            forecasts.append({
                "date": current["date"].iloc[0],
                "model": name,
                "y_true": current[target_col].iloc[0],
                "y_pred": float(prediction),
            })

    return pd.DataFrame(forecasts)

def get_ml_models():
    """Define the two forecasting models and their fixed settings."""
    ridge = GridSearchCV(
        estimator=make_pipeline(
            StandardScaler(),
            Ridge(),
        ),
        param_grid={"ridge__alpha": [10, 100, 1000]},
        cv=TimeSeriesSplit(n_splits=3, test_size=12),
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=1,
        error_score="raise",
    )

    lightgbm = LGBMRegressor(
        n_estimators=60,
        learning_rate=0.05,
        max_depth=2,
        num_leaves=4,
        min_child_samples=15,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )

    return {
    "ridge": ridge,
    "lightgbm": lightgbm,
    "random_forest": RandomForestRegressor(
        n_estimators=80,
        max_depth=3,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    ),
}

def rolling_ml_forecasts(
    data,
    feature_sets,
    evaluation_start,
    train_window=120,
    target_col="target_spread_change",
    return_diagnostics=False,
):
    """Generate rolling forecasts and optionally retain model diagnostics."""
    df = data.sort_values("date").reset_index(drop=True).copy()
    evaluation_start = pd.Timestamp(evaluation_start)
    evaluation_indices = df.index[df["date"] >= evaluation_start]

    if len(evaluation_indices) == 0:
        raise ValueError("No observations in the evaluation period.")

    if evaluation_indices[0] < train_window:
        raise ValueError("Insufficient history for the first forecast.")

    models = get_ml_models()
    forecasts = []
    diagnostics = []

    for feature_set, columns in feature_sets.items():
        for model_name, estimator in models.items():
            print(f"Running {model_name} | {feature_set}")

            for i in evaluation_indices:
                train = df.iloc[i - train_window:i]
                current = df.iloc[[i]]
                date = current["date"].iloc[0]

                fitted = clone(estimator)
                fitted.fit(train[columns], train[target_col])

                prediction = fitted.predict(current[columns])[0]

                forecasts.append({
                    "date": date,
                    "feature_set": feature_set,
                    "model": model_name,
                    "y_true": current[target_col].iloc[0],
                    "y_pred": float(prediction),
                    "ridge_alpha": (
                        fitted.best_params_["ridge__alpha"]
                        if model_name == "ridge"
                        else float("nan")
                    ),
                })

                # Interpret the economically motivated two-region specification
                if (
                    return_diagnostics
                    and feature_set == "prices_both_regions"
                    and model_name in ("ridge", "lightgbm")
                ):
                    if model_name == "ridge":
                        values = (
                            fitted.best_estimator_
                            .named_steps["ridge"]
                            .coef_
                        )
                    else:
                        values = fitted.booster_.feature_importance(
                            importance_type="gain"
                        ).astype(float)

                        # Express gain as a percentage within each fitted model
                        if values.sum() > 0:
                            values = 100 * values / values.sum()

                    diagnostics.extend([
                        {
                            "date": date,
                            "feature_set": feature_set,
                            "model": model_name,
                            "feature": feature,
                            "value": float(value),
                        }
                        for feature, value in zip(columns, values)
                    ])

    predictions = pd.DataFrame(forecasts)

    if return_diagnostics:
        return predictions, pd.DataFrame(diagnostics)

    return predictions

def forecast_metrics(predictions):
    """Evaluate forecasts against their observed outcomes."""
    y_true = predictions["y_true"].to_numpy(dtype=float)
    y_pred = predictions["y_pred"].to_numpy(dtype=float)
    errors = y_true - y_pred

    zero_loss = np.sum(y_true ** 2)
    active = y_pred != 0

    correlation = (
        np.corrcoef(y_true, y_pred)[0, 1]
        if np.std(y_true) > 0 and np.std(y_pred) > 0
        else np.nan
    )

    return {
        "n_forecasts": len(predictions),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "mae": np.mean(np.abs(errors)),
        "r2_zero": (
            1 - np.sum(errors ** 2) / zero_loss
            if zero_loss > 0 else np.nan
        ),
        "correlation": correlation,
        "directional_accuracy": (
            np.mean(np.sign(y_true[active]) == np.sign(y_pred[active]))
            if active.any() else np.nan
        ),
        "signal_coverage": active.mean(),
    }

def evaluate_forecasts(predictions):
    """Compare configurations using identical dates and outcomes."""
    keys = ["model", "feature_set", "date"]

    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate forecasts within a configuration.")

    if predictions[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Missing forecasts or observed outcomes.")

    if predictions.groupby("date")["y_true"].nunique().gt(1).any():
        raise ValueError("Observed outcomes differ across configurations.")

    reference_dates = set(predictions["date"])
    results = []

    for (model, feature_set), group in predictions.groupby(
        ["model", "feature_set"]
    ):
        if set(group["date"]) != reference_dates:
            raise ValueError(
                f"Evaluation dates differ: {model}, {feature_set}"
            )

        results.append({
            "model": model,
            "feature_set": feature_set,
            **forecast_metrics(group),
        })

    return pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)