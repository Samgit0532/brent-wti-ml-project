from pathlib import Path
import pandas as pd

def clean_eia_prices(brent_path, wti_path):
    """Clean and align the two FRED price series on common dates."""

    def load_price(path, name):
        df = pd.read_csv(Path(path), na_values=["."])

        if len(df.columns) != 2:
            raise ValueError(f"{path}: expected a date and a price column.")

        df.columns = ["date", name]
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df[name] = pd.to_numeric(df[name], errors="raise")

        if df["date"].isna().any():
            raise ValueError(f"{path}: missing dates.")

        if df["date"].duplicated().any():
            raise ValueError(f"{path}: duplicate dates.")

        return df.dropna(subset=[name]).sort_values("date")

    brent = load_price(brent_path, "Brent")
    wti = load_price(wti_path, "WTI")

    prices = brent.merge(wti, on="date", how="inner", validate="one_to_one")

    if prices.empty:
        raise ValueError("No common dates with valid Brent and WTI prices.")

    return prices.sort_values("date").reset_index(drop=True)

def clean_jodi_oil(input_dir):
    """Load JODI crude oil fundamentals with consistent units."""

    units = {
        "CLOSTLV": "KBBL",
        "STOCKCH": "KBBL",
        "INDPROD": "KBD",
        "TOTIMPSB": "KBD",
        "TOTEXPSB": "KBD",
        "REFINOBS": "KBD",
    }

    files = sorted(Path(input_dir).glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []

    for path in files:
        df = pd.read_csv(path)

        keep = (
            df["ENERGY_PRODUCT"].eq("CRUDEOIL")
            & df["UNIT_MEASURE"].eq(df["FLOW_BREAKDOWN"].map(units))
        )
        frames.append(df.loc[keep].copy())

    jodi = pd.concat(frames, ignore_index=True).drop_duplicates()

    jodi["TIME_PERIOD"] = pd.to_datetime(
        jodi["TIME_PERIOD"], format="%Y-%m", errors="raise"
    )
    jodi["OBS_VALUE"] = pd.to_numeric(
        jodi["OBS_VALUE"], errors="coerce"
    )

    jodi = jodi.rename(
        columns={
            "REF_AREA": "country",
            "TIME_PERIOD": "date",
            "FLOW_BREAKDOWN": "variable",
            "UNIT_MEASURE": "unit",
            "OBS_VALUE": "value",
            "ASSESSMENT_CODE": "assessment_code",
        }
    )

    keys = ["country", "date", "variable"]

    if jodi.duplicated(keys).any():
        raise ValueError(
            "Multiple records found for the same country, month and variable."
        )

    return (
        jodi[
            ["country", "date", "variable", "unit", "value", "assessment_code"]
        ]
        .sort_values(keys)
        .reset_index(drop=True)
    )