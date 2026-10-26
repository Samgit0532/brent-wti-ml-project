import pandas as pd

def build_monthly_price_features(prices):
    """Construct month-end prices, four predictors and next-month target."""
    df = prices[["date", "Brent", "WTI"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    monthly = (
        df.groupby(df["date"].dt.to_period("M"))
        .tail(1)
        .rename(columns={"date": "price_date"})
        .reset_index(drop=True)
    )

    monthly["date"] = (
        monthly["price_date"].dt.to_period("M").dt.to_timestamp("M")
    )

    current_month = pd.Timestamp.today().to_period("M")
    monthly = monthly.loc[
        monthly["date"].dt.to_period("M") < current_month
    ]

    monthly = monthly.set_index("date").asfreq("ME")

    monthly["spread"] = monthly["Brent"] - monthly["WTI"]
    monthly["spread_change"] = monthly["spread"].diff()
    monthly["spread_change_mean_6"] = (
        monthly["spread_change"].rolling(6).mean()
    )
    monthly["spread_change_vol_12"] = (
        monthly["spread_change"].rolling(12).std()
    )
    monthly["target_spread_change"] = (
        monthly["spread"].shift(-1) - monthly["spread"]
    )

    return monthly.reset_index()

def build_jodi_aggregates(
    jodi,
    benchmark_countries,
    complete_country_pools,
    flows,
):
    """Sum JODI fundamentals over fixed regional and pooled country groups."""
    months = pd.date_range(
        start=jodi["date"].min(),
        end=jodi["date"].max(),
        freq="MS",
    )
    dataset = pd.DataFrame(index=months)
    dataset.index.name = "date"

    for variable in flows:
        panel = (
            jodi.loc[jodi["variable"].eq(variable)]
            .pivot(index="date", columns="country", values="value")
            .reindex(months)
        )

        groups = {
            **benchmark_countries,
            "Pooled": complete_country_pools[variable],
        }

        for group, countries in groups.items():
            if not countries:
                raise ValueError(f"No countries selected: {group}, {variable}")

            selected = panel.reindex(columns=countries)

            if selected.isna().any().any():
                raise ValueError(
                    f"Missing observations: {group}, {variable}"
                )

            dataset[f"{group}_{variable}"] = selected.sum(
                axis=1, min_count=len(countries)
            )

    return dataset.reset_index()

def build_model_dataset(monthly_prices, jodi_dataset, jodi_lag=2):

    """Combine monthly prices with publication-lagged JODI aggregates."""
    fundamentals = jodi_dataset.rename(
        columns={"date": "jodi_reference_month"}
    ).copy()

    fundamentals["date"] = (
        fundamentals["jodi_reference_month"].dt.to_period("M") + jodi_lag
    ).dt.to_timestamp("M")

    dataset = monthly_prices.merge(
        fundamentals,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    # Remove rows without complete predictors or an observed target
    dataset = (
        dataset.dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if dataset.empty:
        raise ValueError("No complete observations after merging.")

    return dataset

def transform_jodi_fundamentals(jodi_dataset):
    """Compute annual growth rates and retain monthly stock changes."""
    data = (
        jodi_dataset.copy()
        .sort_values("date")
        .set_index("date")
        .asfreq("MS")
    )

    transformed = pd.DataFrame(index=data.index)

    for column in data.columns:
        if column.endswith("_STOCKCH"):
            transformed[column] = data[column]
        else:
            previous_year = data[column].shift(12)

            # Growth is undefined when the previous year's value is zero
            denominator = previous_year.where(previous_year.ne(0))

            transformed[f"{column}_yoy"] = (
                data[column].div(denominator).sub(1).mul(100)
            )

    return transformed.reset_index()