from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor


class NaiveZeroBenchmark(BaseEstimator, RegressorMixin):
    r"""
    Random Walk / Efficient Market Benchmark for Spread Variation.
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
    Regresses next-month spread change on the most recent spread change lag:
    \Delta S_{t+1} = \alpha + \beta \Delta S_t + \epsilon_{t+1}.
    """
    def __init__(self, lag_feature_col="spread_change_lag_1"):
        self.lag_feature_col = lag_feature_col
        self.model_ = LinearRegression()

    def _extract_lag_feature(self, X):
        if isinstance(X, pd.DataFrame):
            if self.lag_feature_col in X.columns:
                return X[[self.lag_feature_col]].values
            # Fallback to the first column if the specified column is absent
            return X.iloc[:, [0]].values
        X_arr = np.asarray(X)
        return X_arr[:, [0]]

    def fit(self, X, y):
        x_lag = self._extract_lag_feature(X)
        y_arr = np.asarray(y)

        mask = ~np.isnan(x_lag.ravel()) & ~np.isnan(y_arr)
        if np.sum(mask) >= 5:
            self.model_.fit(x_lag[mask], y_arr[mask])
        else:
            self.model_.intercept_ = 0.0
            self.model_.coef_ = np.array([0.0])
        return self

    def predict(self, X):
        x_lag = self._extract_lag_feature(X)
        return self.model_.predict(x_lag)



class Mbj(BaseEstimator, RegressorMixin):
    r"""
    Britten-Jones (1999) Optimal Weighting Estimator.
    Computes optimal combination weights by regressing a vector of ones (or realized targets)
    on base strategy returns / predictions:
        1 = \sum w_k R_k + \epsilon
    
    Parameters
    ----------
    positive : bool, default=False
        If True, enforces non-negative combination weights (w_k >= 0) via Non-Negative
        Least Squares (NNLS), preventing noisy out-of-sample over-leveraging.
    normalize : bool, default=True
        Whether weights sum to 1.
    """
    def __init__(self, positive=False, normalize=True):
        self.positive = positive
        self.normalize = normalize
        self.coef_ = None

    def fit(self, X, y=None):
        X_mat = np.asarray(X, dtype=float)
        n_samples, n_features = X_mat.shape

        # Britten-Jones target is a constant unit vector (1)
        target = np.ones(n_samples) if y is None else np.asarray(y, dtype=float)

        if self.positive:
            # Solve non-negative least squares: min ||X w - target||_2, s.t. w >= 0
            w, _ = nnls(X_mat, target)
        else:
            # Unconstrained OLS via pseudo-inverse to handle rank deficiency
            w, _, _, _ = np.linalg.lstsq(X_mat, target, rcond=None)

        if self.normalize and np.sum(w) > 1e-12:
            w = w / np.sum(w)
        elif np.sum(w) <= 1e-12:
            # Fallback to equal weighting if all weights collapse to 0
            w = np.full(n_features, 1.0 / n_features)

        self.coef_ = w
        return self

    def predict(self, X):
        X_mat = np.asarray(X, dtype=float)
        return np.dot(X_mat, self.coef_)




def get_models():
    r"""Instantiate forecasting models:

    - Econometric baselines: Naive Zero, Historical Mean, AR(1)
    - Regularized linear model: RidgeCV using instant Generalized CV (cv=None)
    - Non-linear ensemble models: Random Forest, LightGBM
    """
    # Course Ch. 7.1: Regularization grid from 10 to 10000
    alphas_grid = np.exp(np.linspace(np.log(10), np.log(10000), 20))

    models = {
        # Econometric Benchmarks (Instantaneous)
        "naive_zero": NaiveZeroBenchmark(),
        "historical_mean": HistoricalMeanBenchmark(),
        "ar1": AR1Benchmark(lag_feature_col="spread_change_lag_1"),
        # RidgeCV 
        "ridge_cv": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=alphas_grid, cv=None),
        ),
        # Random Forest
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=80,
                max_depth=3,
                min_samples_leaf=8,
                max_features="sqrt",
                random_state=42,
                n_jobs=1,
            ),
        ),
        # LightGBM
        "lightgbm": make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                n_estimators=60,
                learning_rate=0.05,
                max_depth=2,
                num_leaves=4,
                min_child_samples=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1,
                verbose=-1,
            ),
        ),
    }

    return models


def rolling_forecast_with_train_metrics(
    data,
    feature_cols,
    target_col,
    model,
    train_window=120,
    test_size=1,
):
    """
    Rolling-window one-step-ahead forecasting.
    Logs out-of-sample predictions, in-sample training performance,
    and extracts internal parameters (such as RidgeCV chosen alpha).
    """
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    train_predictions = []
    oos_predictions = []
    alphas_selected = []

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

        # Inspect if RidgeCV is present in the pipeline to track alpha over time
        if hasattr(fitted_model, "named_steps") and "ridgecv" in fitted_model.named_steps:
            best_alpha = fitted_model.named_steps["ridgecv"].alpha_
            alphas_selected.append({"date": test["date"].iloc[0], "alpha": best_alpha})

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
    alphas_df = pd.DataFrame(alphas_selected) if len(alphas_selected) > 0 else None

    return train_predictions, oos_predictions, alphas_df


def forecast_metrics(predictions):
    """
    Compute Root Mean Squared Error, Mean Absolute Error, Out-of-Sample R2,
    Pearson Correlation, and Directional Hit Accuracy safely.
    """
    y_true = np.asarray(predictions["y_true"], dtype=float)
    y_pred = np.asarray(predictions["y_pred"], dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    std_true = np.std(y_true)
    std_pred = np.std(y_pred)

    if std_true < 1e-12 or std_pred < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    directional_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": corr,
        "directional_accuracy": directional_accuracy,
    }


def run_overfitting_diagnostics(
    data,
    feature_sets,
    target_col,
    models=None,
    train_windows=(60, 84, 120, 180),
    test_size=1,
):
    """
    Execute rolling-window walk-forward validation across all model architectures,
    information sets, and estimation window sizes.
    """
    if models is None:
        models = get_models()

    diagnostics_results = []
    diagnostics_predictions = {}
    alphas_history = {}

    for train_window in train_windows:
        for feature_set_name, feature_cols in feature_sets.items():
            available_features = [col for col in feature_cols if col in data.columns]

            for model_name, model in models.items():
                print(
                    f"Running {model_name:16s} | {feature_set_name:25s} | "
                    f"train_window={train_window}"
                )

                train_preds, oos_preds, alphas_df = rolling_forecast_with_train_metrics(
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

                if alphas_df is not None:
                    alphas_history[key] = alphas_df

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
    return diagnostics_df, diagnostics_predictions, alphas_history