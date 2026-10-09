"""
scripts/features.py

Feature engineering module for the Brent-WTI spread forecasting project.
Constructs stationary price predictors and transforms JODI fundamentals using
Year-over-Year (YoY) transformations to eliminate seasonality and unit roots.
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
    Decision date is observed at the first calendar day of each month (MS).
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df["Brent"] = pd.to_numeric(df["Brent"], errors="coerce")
    df["WTI"] = pd.to_numeric(df["WTI"], errors="coerce")
    df = df.dropna(subset=["Brent", "WTI"])

    # Monthly calendar grid on month starts
    month_starts = pd.date_range(
        start=df.index.min().to_period("M").to_timestamp(),
        end=df.index.max().to_period("M").to_timestamp(),
        freq="MS",
    )

    # Forward-fill to capture the latest available trading price on or before month start
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

    # Forecasting target: next-month change in the Brent-WTI spread
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
    Transform raw JODI non-stationary series into stationary features without MoM noise.

    Transformations:
    1. Year-over-Year (YoY) percentage changes (12-month lag) for levels:
       CLOSTLV (stocks), INDPROD (production), REFINOBS (refining),
       TOTIMPSB (imports), TOTEXPSB (exports).
       -> Removes unit-root drift and cancels oil market seasonality.
    2. Monthly Stock Changes (STOCKCH):
       Preserved as flow levels (already centered around zero) and first differences.

    Note: Month-over-Month (MoM) transforms are intentionally excluded to prevent
    high-frequency seasonal noise and reduce unnecessary feature dimensionality.
    """
    df = jodi.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "date"]
    data_dict = {"date": df["date"]}

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")

        if keep_raw_levels:
            data_dict[col] = series

        # Handle zero-production edge case (e.g. BE_INDPROD = 0 everywhere)
        if (series.fillna(0) == 0).all():
            data_dict[f"{col}_yoy"] = pd.Series(0.0, index=df.index)
            continue

        # Flow levels and capacities -> YoY percentage change only
        if any(flow in col for flow in ["CLOSTLV", "INDPROD", "REFINOBS", "TOTIMPSB", "TOTEXPSB"]):
            pct_yoy = series.pct_change(12)
            pct_yoy = pct_yoy.mask((series == 0) & (series.shift(12) == 0), 0.0)
            data_dict[f"{col}_yoy"] = pct_yoy

        # Net stock change flow (already a difference)
        elif "STOCKCH" in col:
            data_dict[col] = series

        else:
            data_dict[f"{col}_yoy"] = series.pct_change(12)

    # Construct complete DataFrame in one single memory block (no fragmentation warning)
    transformed = pd.DataFrame(data_dict)
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
    Merge monthly price features with publication-lagged stationary JODI fundamentals.
    """
    prices = monthly_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    jodi_df = jodi.copy()
    jodi_df["date"] = pd.to_datetime(jodi_df["date"])

    if stationarize:
        jodi_df = transform_jodi_fundamentals(jodi_df, keep_raw_levels=keep_raw_levels)

    # Shift by 2 months to strictly eliminate look-ahead publication bias
    jodi_features = jodi_df.set_index("date").shift(jodi_lag).reset_index()

    model_data = prices.merge(jodi_features, on="date", how="left")
    model_data = model_data.sort_values("date").reset_index(drop=True)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model_data.to_csv(p, index=False)

    return model_data