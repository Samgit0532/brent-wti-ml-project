import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor


def get_models():
    r"""
    Define all forecasting models:
    - Econometric baselines (Naive Zero, Historical Mean, AR(1))
    - Regularized linear model (Ridge)
    - Non-linear ensemble models (Random Forest, LightGBM)
    """
    models = {
        # Econometric Benchmarks
        "naive_zero": NaiveZeroBenchmark(),
        "historical_mean": HistoricalMeanBenchmark(),
        "ar1": AR1Benchmark(lag_feature_col="spread_change"),

        # Machine Learning Models
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
                n_jobs=1,  # n_jobs=1 prevents Windows multiprocessing deadlocks
            ),
        ),

        "lightgbm": make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=2,
                num_leaves=4,
                min_child_samples=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,  # n_jobs=1 prevents Windows multiprocessing deadlocks
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
    Compute forecasting performance metrics safely,
    handling zero-variance predictions (e.g., naive zero benchmark).
    """
    y_true = np.asarray(predictions["y_true"], dtype=float)
    y_pred = np.asarray(predictions["y_pred"], dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Safe correlation calculation: avoids division by zero if std is zero
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)

    if std_true < 1e-12 or std_pred < 1e-12:
        corr = 0.0
    else:
        # np.corrcoef returns a 2x2 matrix
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    # Directional accuracy: sign agreement
    directional_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": corr,
        "directional_accuracy": directional_accuracy,
    }

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



class NaiveZeroBenchmark(BaseEstimator, RegressorMixin):
    r"""
    Random Walk / No-Change Benchmark for Spread Variation.
    Always forecasts zero change: \hat{\Delta S}_{t+1} = 0.
    """
    def fit(self, X, y):
        return self

    def predict(self, X):
        n_samples = len(X)
        return np.zeros(n_samples)


class HistoricalMeanBenchmark(BaseEstimator, RegressorMixin):
    r"""
    Prevailing Historical Mean Benchmark.
    Forecasts the expanding historical average of target changes:
    \hat{\Delta S}_{t+1} = (1 / t) \sum_{\tau=1}^t \Delta S_\tau.
    """
    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.mean_ = float(np.nanmean(y_arr))
        return self

    def predict(self, X):
        n_samples = len(X)
        return np.full(n_samples, fill_value=self.mean_)


class AR1Benchmark(BaseEstimator, RegressorMixin):
    r"""
    Univariate Autoregressive AR(1) Benchmark.
    Regresses the next-month spread change on the most recent spread change:
    \Delta S_{t+1} = \alpha + \beta \Delta S_t + \epsilon_{t+1}.
    """
    def __init__(self, lag_feature_col="spread_change"):
        self.lag_feature_col = lag_feature_col
        self.model_ = LinearRegression()

    def _extract_lag_feature(self, X):
        if isinstance(X, pd.DataFrame):
            if self.lag_feature_col in X.columns:
                return X[[self.lag_feature_col]].values
            # Fallback to the first column if named column is missing
            return X.iloc[:, [0]].values
        X_arr = np.asarray(X)
        return X_arr[:, [0]]

    def fit(self, X, y):
        x_lag = self._extract_lag_feature(X)
        y_arr = np.asarray(y)

        # Drop any NaNs before fitting
        mask = ~np.isnan(x_lag.ravel()) & ~np.isnan(y_arr)
        if np.sum(mask) >= 3:
            self.model_.fit(x_lag[mask], y_arr[mask])
        else:
            self.model_.intercept_ = 0.0
            self.model_.coef_ = np.array([0.0])
        return self

    def predict(self, X):
        x_lag = self._extract_lag_feature(X)
        return self.model_.predict(x_lag)


def diebold_mariano_test(y_true, y_pred1, y_pred2, h=None, loss_type="sq", use_hln=True):
    r"""
    Diebold-Mariano (1995) and Harvey-Leybourne-Newbold (HLN, 1997) tests
    for comparing predictive accuracy between two competing forecasts.

    Formulas implemented:
    - Loss differential: d_i = e1_i^2 - e2_i^2 
    - Autocovariances: gamma_k = (1/n) * sum((d_i - mean_d) * (d_{i-k} - mean_d))
    - Bandwidth rule: h = int(n**(1/3)) + 1 
    - Variance of mean loss differential: V(mean_d) = (1/n) * [gamma_0 + 2 * sum_{k=1}^{h-1} gamma_k]
    - Standard DM statistic: DM = mean_d / sqrt(V(mean_d)) ~ N(0, 1)
    - HLN small-sample adjusted statistic:
        DM* = DM * sqrt([n + 1 - 2*h + h*(h - 1)/n] / n) ~ Student-t(n - 1)

    Parameters
    ----------
    y_true : array-like
        Ground truth realized values (y_i).
    y_pred1 : array-like
        Forecasts from Model 1 (f_i).
    y_pred2 : array-like
        Forecasts from Model 2 (g_i), e.g., Naive zero benchmark.
    h : int, optional
        Truncation lag. If None, sets h = int(n**(1/3)) + 1 as per standard literature.
    loss_type : str, default='sq'
        'sq' for squared error loss (MSE), 'abs' for absolute error loss (MAE).
    use_hln : bool, default=True
        Whether to apply the Harvey-Leybourne-Newbold small-sample correction.

    Returns
    -------
    stat : float
        The test statistic (DM or HLN adjusted DM*).
        Negative value indicates Model 1 has lower loss (is better) than Model 2.
    p_value : float
        Two-tailed p-value (from Student-t if use_hln=True, else from Standard Normal).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred1 = np.asarray(y_pred1, dtype=float)
    y_pred2 = np.asarray(y_pred2, dtype=float)

    # 1. Forecast residuals (e_i and r_i)
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    # 2. Loss differential series (d_i)
    if loss_type == "sq":
        d = (e1 ** 2) - (e2 ** 2)
    elif loss_type == "abs":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss_type must be either 'sq' or 'abs'")

    n = len(d)
    if n < 3:
        return 0.0, 1.0

    mean_d = np.mean(d)

    # 3. Automatic bandwidth selection: h = int(n^(1/3)) + 1
    if h is None:
        h = int(np.floor(n ** (1.0 / 3.0))) + 1
    h = max(1, min(h, n - 1))

    # 4. Autocovariances gamma_k for k = 0, ..., h-1
    # gamma_0 is sample variance with divisor n (VAR.P)
    d_centered = d - mean_d
    gamma_0 = np.mean(d_centered ** 2)

    sum_cov = 0.0
    for k in range(1, h):
        gamma_k = np.mean(d_centered[k:] * d_centered[:-k])
        sum_cov += 2.0 * gamma_k

    # Variance of the mean differential
    var_d = (gamma_0 + sum_cov) / n

    if var_d <= 1e-12:
        return 0.0, 1.0

    # 5. Standard Diebold-Mariano statistic
    dm_stat = mean_d / np.sqrt(var_d)

    # 6. Small-sample correction: Harvey, Leybourne, and Newbold (HLN, 1997)
    if use_hln:
        hln_factor = np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n)
        stat = dm_stat * hln_factor
        # Student-t distribution with (n - 1) degrees of freedom
        df = n - 1
        p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(stat), df=df))
    else:
        stat = dm_stat
        # Standard Normal distribution N(0, 1)
        p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat)))

    return float(stat), float(p_value)
