import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_summary(data, columns):
    """ADF tests with an intercept and AIC-based lag selection."""
    results = []

    for column in columns:
        series = data[column].dropna()

        if series.nunique() <= 1:
            raise ValueError(f"Cannot test a constant series: {column}")

        statistic, p_value, lags, nobs, *_ = adfuller(
            series,
            regression="c",
            autolag="AIC",
            result_object=False,
        )

        results.append({
            "variable": column,
            "ADF statistic": statistic,
            "p-value": p_value,
            "lags": lags,
            "observations": nobs,
            "reject unit root (5%)": p_value < 0.05,
        })

    return pd.DataFrame(results).set_index("variable")