from pathlib import Path
import numpy as np
import pandas as pd


def build_monthly_price_features(
    prices,
    output_path=None,
    lags=(1, 2, 3),
    rolling_mean_windows=(6, 12),
    rolling_vol_windows=(3, 12),
    annualization=12,
):
    """
    Build monthly Brent-WTI price features from cleaned daily spot prices.

    The monthly price is observed at the first day of each month. If the
    first day of the month is not available in the daily price data, the
    function uses the last available trading-day price before that date.

    The function creates short-term lagged features, medium-term rolling
    mean features, rolling volatility features, rolling correlation features,
    and the next-month spread change target.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned daily price dataframe containing the columns:
        - date
        - Brent
        - WTI

    output_path : str or Path, optional
        Path where the monthly feature dataset should be saved.

    lags : tuple of int, default=(1, 2, 3)
        Lags used to create short-term price features.

    rolling_mean_windows : tuple of int, default=(6, 12)
        Rolling windows, in months, used to compute medium-term trend features.

    rolling_vol_windows : tuple of int, default=(3, 12)
        Rolling windows, in months, used to compute realized volatility features.

    annualization : int, default=12
        Annualization factor used for monthly volatility.

    Returns
    -------
    pd.DataFrame
        Monthly dataframe containing price features and the next-month target.
    """

    prices = prices.copy()

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").set_index("date")

    prices["Brent"] = pd.to_numeric(prices["Brent"], errors="coerce")
    prices["WTI"] = pd.to_numeric(prices["WTI"], errors="coerce")

    prices = prices.dropna(subset=["Brent", "WTI"])

    # Monthly decision dates: first calendar day of each month
    month_starts = pd.date_range(
        start=prices.index.min().to_period("M").to_timestamp(),
        end=prices.index.max().to_period("M").to_timestamp(),
        freq="MS",
    )

    # Use the last available trading price before or on each month start
    monthly_prices = (
        prices.reindex(prices.index.union(month_starts))
        .sort_index()
        .ffill()
        .loc[month_starts]
        .dropna()
    )

    monthly = monthly_prices.copy()

    # Core price variables
    monthly["spread"] = monthly["Brent"] - monthly["WTI"]

    monthly["brent_return"] = monthly["Brent"].pct_change()
    monthly["wti_return"] = monthly["WTI"].pct_change()

    monthly["spread_change"] = monthly["spread"].diff()
    monthly["spread_return"] = monthly["brent_return"] - monthly["wti_return"]

    # Short-term lagged features
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

    # Medium-term rolling mean features
    features_to_average = [
        "spread_change",
        "brent_return",
        "wti_return",
        "spread_return",
    ]

    for window in rolling_mean_windows:
        for col in features_to_average:
            monthly[f"{col}_rolling_mean_{window}"] = (
                monthly[col]
                .rolling(window=window)
                .mean()
            )

    # Rolling monthly volatility features
    return_features = [
        "brent_return",
        "wti_return",
        "spread_return",
    ]

    for window in rolling_vol_windows:
        for col in return_features:
            monthly[f"{col}_rolling_vol_{window}"] = (
                monthly[col]
                .rolling(window=window)
                .std()
                * np.sqrt(annualization)
            )

    # Rolling Brent-WTI correlation
    monthly["brent_wti_corr_12"] = (
        monthly["brent_return"]
        .rolling(window=12)
        .corr(monthly["wti_return"])
    )

    # Main prediction target: next-month change in the Brent-WTI spread
    monthly["target_spread_change_next_month"] = monthly["spread_change"].shift(-1)

    monthly = monthly.reset_index().rename(columns={"index": "date"})

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(output_path, index=False)

    return monthly

def build_model_dataset(
    monthly_prices,
    jodi,
    output_path,
    jodi_lag=1,
):
    """
    Build the final monthly modelling dataset by merging price features with JODI fundamentals.

    JODI variables are lagged before the merge in order to reduce look-ahead bias.
    For example, with jodi_lag=1, JODI fundamentals dated at month t are only used
    from month t+1 onward.

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        Monthly price feature dataframe containing:
        - date
        - Brent
        - WTI
        - spread
        - price-based features
        - target_spread_change_next_month

    jodi : pd.DataFrame
        Cleaned JODI dataframe containing:
        - date
        - country-flow fundamental variables

    output_path : str or Path
        Path where the final modelling dataset should be saved.

    jodi_lag : int, default=1
        Number of months by which JODI variables are lagged.

    Returns
    -------
    pd.DataFrame
        Final modelling dataframe containing price features, lagged JODI features,
        and the prediction target.
    """

    monthly_prices = monthly_prices.copy()
    jodi = jodi.copy()

    monthly_prices["date"] = pd.to_datetime(monthly_prices["date"])
    jodi["date"] = pd.to_datetime(jodi["date"])

    monthly_prices = monthly_prices.sort_values("date").reset_index(drop=True)
    jodi = jodi.sort_values("date").reset_index(drop=True)

    jodi_features = jodi.set_index("date")

    # Lag JODI features to avoid look-ahead bias
    jodi_features = jodi_features.shift(jodi_lag)

    jodi_features = jodi_features.reset_index()

    model_data = monthly_prices.merge(
        jodi_features,
        on="date",
        how="left",
    )

    model_data = model_data.sort_values("date").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(output_path, index=False)

    return model_data