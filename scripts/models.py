import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor


def get_models():
    """
    Define efficient forecasting models used in the project.

    Returns
    -------
    dict
        Dictionary mapping model names to sklearn-compatible estimators.
    """

    models = {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),

        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=8,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),

        "lightgbm": make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                num_leaves=4,
                min_child_samples=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
        ),
    }

    return models


def rolling_forecast(
    data,
    feature_cols,
    target_col,
    model,
    train_window=120,
    test_size=1,
):
    """
    Run a rolling-window one-step-ahead forecasting procedure.

    At each step, the model is trained on a fixed-size historical window
    and then used to predict the next observation.

    Parameters
    ----------
    data : pd.DataFrame
        Modelling dataset containing:
        - date
        - feature columns
        - target column

    feature_cols : list of str
        Names of the columns used as predictors.

    target_col : str
        Name of the target column.

    model : sklearn-compatible estimator
        Model used for prediction.

    train_window : int, default=120
        Number of monthly observations used for training.

    test_size : int, default=1
        Number of observations predicted at each step.

    Returns
    -------
    pd.DataFrame
        DataFrame containing dates, realized values, and predictions.
    """

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    predictions = []

    for start in range(0, len(df) - train_window - test_size + 1, test_size):
        train_start = start
        train_end = start + train_window
        test_end = train_end + test_size

        train = df.iloc[train_start:train_end]
        test = df.iloc[train_end:test_end]

        X_train = train[feature_cols]
        y_train = train[target_col]

        X_test = test[feature_cols]
        y_test = test[target_col]

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        y_pred = fitted_model.predict(X_test)

        pred_df = pd.DataFrame(
            {
                "date": test["date"].values,
                "y_true": y_test.values,
                "y_pred": y_pred,
            }
        )

        predictions.append(pred_df)

    predictions = pd.concat(predictions, ignore_index=True)

    return predictions


def forecast_metrics(predictions):
    """
    Compute forecasting performance metrics.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame containing:
        - y_true
        - y_pred

    Returns
    -------
    dict
        Forecasting performance metrics.
    """

    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    corr = pd.Series(y_true).corr(pd.Series(y_pred))

    directional_accuracy = (
        np.sign(y_true) == np.sign(y_pred)
    ).mean()

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": corr,
        "directional_accuracy": directional_accuracy,
    }

    return metrics


def run_model_comparison(
    data,
    feature_sets,
    target_col,
    models=None,
    train_window=120,
    test_size=1,
):
    """
    Run rolling-window forecasts for several models and feature sets.

    Parameters
    ----------
    data : pd.DataFrame
        Final modelling dataset.

    feature_sets : dict
        Dictionary mapping feature set names to lists of feature columns.

    target_col : str
        Name of the target column.

    models : dict, optional
        Dictionary mapping model names to sklearn-compatible estimators.
        If None, default models are used.

    train_window : int, default=120
        Number of monthly observations used for training.

    test_size : int, default=1
        Number of observations predicted at each step.

    Returns
    -------
    tuple
        predictions_dict : dict
            Dictionary containing predictions for each model and feature set.

        metrics_df : pd.DataFrame
            Summary table of forecasting metrics.
    """

    if models is None:
        models = get_models()

    predictions_dict = {}
    metrics_list = []

    for feature_set_name, feature_cols in feature_sets.items():
        available_features = [col for col in feature_cols if col in data.columns]

        for model_name, model in models.items():
            key = f"{feature_set_name}__{model_name}"

            preds = rolling_forecast(
                data=data,
                feature_cols=available_features,
                target_col=target_col,
                model=model,
                train_window=train_window,
                test_size=test_size,
            )

            metrics = forecast_metrics(preds)
            metrics["feature_set"] = feature_set_name
            metrics["model"] = model_name
            metrics["n_features"] = len(available_features)

            predictions_dict[key] = preds
            metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    metrics_df = metrics_df[
        [
            "feature_set",
            "model",
            "n_features",
            "rmse",
            "mae",
            "r2",
            "correlation",
            "directional_accuracy",
        ]
    ]

    return predictions_dict, metrics_df


def rolling_forecast_with_train_metrics(
    data,
    feature_cols,
    target_col,
    model,
    train_window=120,
    test_size=1,
):
    """
    Run rolling-window forecasts and store both train and out-of-sample predictions.

    This function is used to diagnose overfitting by comparing in-sample
    performance with out-of-sample performance.
    """

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    train_predictions = []
    oos_predictions = []

    for start in range(0, len(df) - train_window - test_size + 1, test_size):
        train_start = start
        train_end = start + train_window
        test_end = train_end + test_size

        train = df.iloc[train_start:train_end]
        test = df.iloc[train_end:test_end]

        X_train = train[feature_cols]
        y_train = train[target_col]

        X_test = test[feature_cols]
        y_test = test[target_col]

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        y_train_pred = fitted_model.predict(X_train)
        y_test_pred = fitted_model.predict(X_test)

        train_pred_df = pd.DataFrame(
            {
                "date": train["date"].values,
                "y_true": y_train.values,
                "y_pred": y_train_pred,
                "forecast_date": test["date"].iloc[0],
            }
        )

        oos_pred_df = pd.DataFrame(
            {
                "date": test["date"].values,
                "y_true": y_test.values,
                "y_pred": y_test_pred,
            }
        )

        train_predictions.append(train_pred_df)
        oos_predictions.append(oos_pred_df)

    train_predictions = pd.concat(train_predictions, ignore_index=True)
    oos_predictions = pd.concat(oos_predictions, ignore_index=True)

    return train_predictions, oos_predictions



def run_overfitting_diagnostics(
    data,
    feature_sets,
    target_col,
    models=None,
    train_windows=(24, 36, 60, 84, 120, 180),
    test_size=1,
):
    """
    Run rolling-window forecasts for several models, feature sets, and training windows,
    and compute both train and out-of-sample metrics.

    Parameters
    ----------
    data : pd.DataFrame
        Final modelling dataset.

    feature_sets : dict
        Dictionary mapping feature set names to feature columns.

    target_col : str
        Name of the target variable.

    models : dict, optional
        Dictionary of sklearn-compatible models.

    train_windows : tuple of int
        Rolling training window lengths.

    test_size : int, default=1
        Number of observations predicted at each step.

    Returns
    -------
    tuple
        diagnostics_df : pd.DataFrame
            Summary table with train and OOS metrics.

        diagnostics_predictions : dict
            Dictionary containing train and OOS predictions for each configuration.
    """

    if models is None:
        models = get_models()

    diagnostics_results = []
    diagnostics_predictions = {}

    for train_window in train_windows:
        for feature_set_name, feature_cols in feature_sets.items():
            available_features = [col for col in feature_cols if col in data.columns]

            for model_name, model in models.items():
                print(
                    f"Running {model_name} | {feature_set_name} | "
                    f"train_window={train_window}"
                )

                train_preds, oos_preds = rolling_forecast_with_train_metrics(
                    data=data,
                    feature_cols=available_features,
                    target_col=target_col,
                    model=model,
                    train_window=train_window,
                    test_size=test_size,
                )

                train_metrics = forecast_metrics(train_preds)
                oos_metrics = forecast_metrics(oos_preds)

                key = f"{feature_set_name}__{model_name}__window_{train_window}"

                diagnostics_predictions[key] = {
                    "train": train_preds,
                    "oos": oos_preds,
                }

                diagnostics_results.append(
                    {
                        "train_window": train_window,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "n_features": len(available_features),
                        "n_oos": len(oos_preds),

                        "train_rmse": train_metrics["rmse"],
                        "oos_rmse": oos_metrics["rmse"],
                        "rmse_gap": oos_metrics["rmse"] - train_metrics["rmse"],

                        "train_mae": train_metrics["mae"],
                        "oos_mae": oos_metrics["mae"],

                        "train_r2": train_metrics["r2"],
                        "oos_r2": oos_metrics["r2"],
                        "r2_gap": train_metrics["r2"] - oos_metrics["r2"],

                        "train_correlation": train_metrics["correlation"],
                        "oos_correlation": oos_metrics["correlation"],

                        "train_directional_accuracy": train_metrics["directional_accuracy"],
                        "oos_directional_accuracy": oos_metrics["directional_accuracy"],

                        "oos_start": oos_preds["date"].min(),
                        "oos_end": oos_preds["date"].max(),
                    }
                )

    diagnostics_df = pd.DataFrame(diagnostics_results)

    return diagnostics_df, diagnostics_predictions