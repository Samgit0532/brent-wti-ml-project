from pathlib import Path
import pandas as pd

def clean_eia_prices(brent_path, wti_path, output_path):
    """
    Clean Brent and WTI EIA spot price CSV files and merge them on common dates.

    Expected format for each raw CSV:
    - observation_date
    - one price column, for example DCOILBRENTEU or DCOILWTICO

    Parameters
    ----------
    brent_path : str or Path
        Path to the raw Brent CSV file.

    wti_path : str or Path
        Path to the raw WTI CSV file.

    output_path : str or Path
        Path where the cleaned merged CSV should be saved.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns:
        - date
        - Brent
        - WTI
    """

    brent_path = Path(brent_path)
    wti_path = Path(wti_path)

    brent = pd.read_csv(brent_path)
    wti = pd.read_csv(wti_path)

    brent_price_col = brent.columns[1]
    wti_price_col = wti.columns[1]

    brent["observation_date"] = pd.to_datetime(brent["observation_date"])
    wti["observation_date"] = pd.to_datetime(wti["observation_date"])

    brent[brent_price_col] = pd.to_numeric(brent[brent_price_col], errors="coerce")
    wti[wti_price_col] = pd.to_numeric(wti[wti_price_col], errors="coerce")

    brent = brent.rename(
        columns={
            "observation_date": "date",
            brent_price_col: "Brent",
        }
    )

    wti = wti.rename(
        columns={
            "observation_date": "date",
            wti_price_col: "WTI",
        }
    )

    brent = brent[["date", "Brent"]].dropna()
    wti = wti[["date", "WTI"]].dropna()

    prices = brent.merge(wti, on="date", how="inner")
    prices = prices.sort_values("date").reset_index(drop=True)

    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)

    return prices

def clean_jodi_oil(
    input_dir,
    output_path,
    countries,
    product="CRUDEOIL",
):
    """
    Clean and merge yearly JODI Oil CSV files into a single wide dataframe.

    Expected format for each raw CSV:
    - REF_AREA
    - TIME_PERIOD
    - ENERGY_PRODUCT
    - FLOW_BREAKDOWN
    - UNIT_MEASURE
    - OBS_VALUE
    - ASSESSMENT_CODE

    Parameters
    ----------
    input_dir : str or Path
        Path to the directory containing yearly JODI CSV files.

    output_path : str or Path
        Path where the cleaned JODI dataset should be saved.

    countries : list of str
        List of country codes to include in the cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned wide dataframe with one row per month and one column per
        country-flow variable.
    """

    input_dir = Path(input_dir)

    flows = [
        "CLOSTLV",   # Closing stock levels
        "STOCKCH",   # Stock changes
        "INDPROD",   # Indigenous production
        "TOTIMPSB",  # Total imports
        "TOTEXPSB",  # Total exports
        "REFINOBS",  # Refinery intake / refinery observations
    ]

    units = {
        "CLOSTLV": "KBBL",
        "STOCKCH": "KBBL",
        "INDPROD": "KBD",
        "TOTIMPSB": "KBD",
        "TOTEXPSB": "KBD",
        "REFINOBS": "KBD",
    }

    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dataframes = []

    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)

    jodi = pd.concat(dataframes, ignore_index=True)

    jodi["date"] = pd.to_datetime(jodi["TIME_PERIOD"])
    jodi["OBS_VALUE"] = pd.to_numeric(jodi["OBS_VALUE"], errors="coerce")

    jodi = jodi[jodi["ENERGY_PRODUCT"] == product]
    jodi = jodi[jodi["FLOW_BREAKDOWN"].isin(flows)]

    if countries is not None:
        jodi = jodi[jodi["REF_AREA"].isin(countries)]

    unit_filter = pd.Series(False, index=jodi.index)

    for flow, unit in units.items():
        unit_filter |= (
            (jodi["FLOW_BREAKDOWN"] == flow)
            & (jodi["UNIT_MEASURE"] == unit)
        )

    jodi = jodi[unit_filter]

    jodi = jodi.dropna(subset=["date", "OBS_VALUE"])

    jodi["variable"] = (
        jodi["REF_AREA"]
        + "_"
        + jodi["FLOW_BREAKDOWN"]
    )

    jodi_wide = jodi.pivot_table(
        index="date",
        columns="variable",
        values="OBS_VALUE",
        aggfunc="mean",
    )

    jodi_wide = jodi_wide.sort_index().reset_index()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jodi_wide.to_csv(output_path, index=False)

    return jodi_wide