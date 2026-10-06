import pandas as pd


def check_missing_values(data, columns=None, start_date=None, end_date=None, top_n=20):
    """
    Check missing values in a dataframe over a given date range.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing a date column.

    columns : list of str, optional
        Columns to check. If None, all columns except date are checked.

    start_date : str, optional
        Start date of the sample.

    end_date : str, optional
        End date of the sample.

    top_n : int, default=20
        Number of columns with the most missing values to display.

    Returns
    -------
    pd.Series
        Number of missing values by column, sorted in decreasing order.
    """

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    if columns is None:
        columns = [col for col in df.columns if col != "date"]

    missing = df[columns].isna().sum().sort_values(ascending=False)

    print(f"Sample period: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of checked columns: {len(columns)}")
    print()
    print(f"Top {top_n} columns with missing values:")

    return missing.head(top_n)

def summarize_model_sample(model_sample, feature_cols, target_col):
    """
    Summarize the modelling sample and report missing values.

    Parameters
    ----------
    model_sample : pd.DataFrame
        Modelling sample containing date, features, and target.

    feature_cols : list of str
        Feature columns used by the model.

    target_col : str
        Target column.

    Returns
    -------
    dict
        Summary statistics about the modelling sample.
    """

    summary = {
        "start_date": model_sample["date"].min(),
        "end_date": model_sample["date"].max(),
        "n_rows": len(model_sample),
        "n_features": len(feature_cols),
        "missing_features": model_sample[feature_cols].isna().sum().sum(),
        "missing_target": model_sample[target_col].isna().sum(),
    }

    return summary

def last_complete_date(data, columns=None):
    """
    Find the last date for which all selected columns are non-missing.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing a date column.

    columns : list of str, optional
        Columns to check. If None, all columns except date are checked.

    Returns
    -------
    pd.Timestamp
        Last date where all selected columns are available.
    """

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if columns is None:
        columns = [col for col in df.columns if col != "date"]

    missing_by_date = df.set_index("date")[columns].isna().sum(axis=1)

    complete_dates = missing_by_date[missing_by_date == 0]

    if complete_dates.empty:
        return None

    return complete_dates.index.max()