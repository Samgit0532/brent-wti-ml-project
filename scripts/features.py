"""
scripts/features.py

Feature engineering module for the Brent-WTI spread forecasting project.
Constructs stationary price features and transforms JODI oil market fundamentals
with realistic publication lags to avoid look-ahead bias.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def build_monthly_price_features(
    prices: pd.DataFrame,
    output_path: Path | str | None = None,
    lags: tuple[int, ...] = (1, 2, 3),
    rolling_mean_windows: tuple[int, ...] = (6, 12),
    rolling_vol_windows: tuple[int, ...] = (3, 12),
    annualization: int = 12,
) -> pd.DataFrame:
    """
    Build monthly price-based features from cleaned daily spot prices.

    The monthly decision point is set to the first calendar day of each month (MS).
    If the first day is a weekend or holiday, the last available trading price 
    on or before that date is used (forward fill).

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned daily price DataFrame with columns: ['date', 'Brent', 'WTI'].
    output_path : Path or str, optional
        File path where the processed CSV will be saved.
    lags : tuple of int, default=(1, 2, 3)
        Autoregressive lag horizons (in months).
    rolling_mean_windows : tuple of int, default=(6, 12)
        Rolling windows (in months) for momentum/trend features.
    rolling_vol_windows : tuple of int, default=(3, 12)
        Rolling windows (in months) for realized volatility estimation.
    annualization : int, default=12
        Scaling factor to annualize monthly standard deviation.

    Returns
    -------
    pd.DataFrame
        Monthly dataset containing price predictors and the forward target variable.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df["Brent"] = pd.to_numeric(df["Brent"], errors="coerce")
    df["WTI"] = pd.to_numeric(df["WTI"], errors="coerce")
    df = df.dropna(subset=["Brent", "WTI"])

    # Create a regular monthly index on the first day of each calendar month
    month_starts = pd.date_range(
        start=df.index.min().to_period("M").to_timestamp(),
        end=df.index.max().to_period("M").to_timestamp(),
        freq="MS",
    )

    # Reindex and forward-fill to capture the latest known price as of month start
    monthly = (
        df.reindex(df.index.union(month_starts))
        .sort_index()
        .ffill()
        .loc[month_starts]
        .dropna()
        .copy()
    )

    # Core spread and return variables
    monthly["spread"] = monthly["Brent"] - monthly["WTI"]
    monthly["brent_return"] = monthly["Brent"].pct_change()
    monthly["wti_return"] = monthly["WTI"].pct_change()
    monthly["spread_change"] = monthly["spread"].diff()
    monthly["spread_return"] = monthly["brent_return"] - monthly["wti_return"]

    # Short-term lagged predictors
    features_to_lag = [
        "spread",
        "spread_change",
        "brent_return",
        "wti_return",
        "spread_return",
    ]
    for lag in lags:
        for col in features_to_lag:
            monthly[f"{col}_lag_{lag}"] = monthly[col].shift(lag)

    # Medium-term rolling moving averages
    features_to_average = [
        "spread_change",
        "brent_return",
        "wti_return",
        "spread_return",
    ]
    for window in rolling_mean_windows:
        for col in features_to_average:
            monthly[f"{col}_rolling_mean_{window}"] = (
                monthly[col].rolling(window=window).mean()
            )

    # Realized rolling volatility features
    vol_targets = ["brent_return", "wti_return", "spread_return"]
    for window in rolling_vol_windows:
        for col in vol_targets:
            monthly[f"{col}_rolling_vol_{window}"] = (
                monthly[col].rolling(window=window).std() * np.sqrt(annualization)
            )

    # Rolling 12-month correlation between Brent and WTI returns
    monthly["brent_wti_corr_12"] = (
        monthly["brent_return"].rolling(window=12).corr(monthly["wti_return"])
    )

    # Primary forecasting target: next-month change in the Brent-WTI spread
    monthly["target_spread_change_next_month"] = monthly["spread_change"].shift(-1)

    monthly = monthly.reset_index().rename(columns={"index": "date"})

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(p, index=False)

    return monthly


def transform_jodi_fundamentals(
    jodi: pd.DataFrame,
    keep_raw_levels: bool = False,
) -> pd.DataFrame:
    """
    Transform raw JODI non-stationary level series into stationary features.
    Safely handles countries with zero production/trade (e.g. Belgium INDPROD = 0).
    """
    df = jodi.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "date"]
    transformed = pd.DataFrame({"date": df["date"]})

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")

        if keep_raw_levels:
            transformed[col] = series

        # If a series is completely zero everywhere (e.g., BE_INDPROD = 0.0)
        if (series.fillna(0) == 0).all():
            transformed[f"{col}_yoy"] = 0.0
            transformed[f"{col}_mom"] = 0.0
            continue

        # Flow variables in levels/rates: stocks, production, intake, trade
        if any(flow in col for flow in ["CLOSTLV", "INDPROD", "REFINOBS", "TOTIMPSB", "TOTEXPSB"]):
            # 12-month percentage change (YoY)
            pct_yoy = series.pct_change(12)
            # If original series didn't change and was 0, fill with 0
            pct_yoy = pct_yoy.mask((series == 0) & (series.shift(12) == 0), 0.0)
            transformed[f"{col}_yoy"] = pct_yoy

            # 1-month percentage change (MoM)
            pct_mom = series.pct_change(1)
            pct_mom = pct_mom.mask((series == 0) & (series.shift(1) == 0), 0.0)
            transformed[f"{col}_mom"] = pct_mom

        # Flow variables representing changes (already differences)
        elif "STOCKCH" in col:
            transformed[f"{col}_diff"] = series.diff(1)
            transformed[col] = series

        else:
            transformed[f"{col}_diff"] = series.diff(1)

    # Clean infinite values caused by divisions by zero
    transformed = transformed.replace([np.inf, -np.inf], np.nan)

    return transformed

def build_model_dataset(
    monthly_prices: pd.DataFrame,
    jodi: pd.DataFrame,
    output_path: Path | str | None = None,
    jodi_lag: int = 2,
    stationarize: bool = True,
    keep_raw_levels: bool = False,
) -> pd.DataFrame:
    """
    Merge monthly price predictors with stationary, publication-lagged JODI fundamentals.

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        Processed monthly price dataset.
    jodi : pd.DataFrame
        Cleaned JODI oil fundamentals dataset.
    output_path : Path or str, optional
        Path where the final merged dataset will be saved.
    jodi_lag : int, default=2
        Number of months to lag JODI fundamentals.
        Default is 2 (M-2) to respect JODI's official 50-60 day publication delay
        and eliminate look-ahead bias.
    stationarize : bool, default=True
        Whether to transform non-stationary fundamental levels into YoY/MoM changes.
    keep_raw_levels : bool, default=False
        Whether to retain raw level series if stationarize is True.

    Returns
    -------
    pd.DataFrame
        Final dataset for out-of-sample forecasting and backtesting.
    """
    prices = monthly_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    jodi_df = jodi.copy()
    jodi_df["date"] = pd.to_datetime(jodi_df["date"])

    # Stationarize fundamentals if requested
    if stationarize:
        jodi_df = transform_jodi_fundamentals(jodi_df, keep_raw_levels=keep_raw_levels)

    # Apply publication lag strictly across all fundamental predictors
    jodi_features = jodi_df.set_index("date").shift(jodi_lag).reset_index()

    # Left join to preserve complete timeline through the end of the sample (up to 2026)
    model_data = prices.merge(jodi_features, on="date", how="left")
    model_data = model_data.sort_values("date").reset_index(drop=True)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model_data.to_csv(p, index=False)

    return model_data