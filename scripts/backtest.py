import numpy as np
import pandas as pd

def build_spread_positions(predictions, monthly_prices):
    """Convert forecasts into equal-barrel positions and gross P&L."""
    market = (
        monthly_prices[["date", "price_date", "Brent", "WTI"]]
        .sort_values("date")
        .copy()
    )

    # Identify the next monthly observation before joining forecasts
    market["realization_date"] = market["date"].shift(-1)
    market["exit_price_date"] = market["price_date"].shift(-1)
    market["next_Brent"] = market["Brent"].shift(-1)
    market["next_WTI"] = market["WTI"].shift(-1)

    trades = predictions.merge(
        market,
        on="date",
        how="left",
        validate="many_to_one",
    )

    required = [
        "y_pred", "y_true", "Brent", "WTI",
        "next_Brent", "next_WTI",
        "price_date", "exit_price_date", "realization_date",
    ]

    if trades[required].isna().any().any():
        raise ValueError("Missing forecasts, outcomes or trading prices.")

    trades["realized_spread_change"] = (
        trades["next_Brent"] - trades["Brent"]
        - trades["next_WTI"] + trades["WTI"]
    )

    if not np.allclose(
        trades["realized_spread_change"],
        trades["y_true"],
    ):
        raise ValueError("Forecast outcomes and trading dates do not match.")

    trades["position"] = np.sign(trades["y_pred"]).astype(int)
    trades["brent_quantity"] = trades["position"]
    trades["wti_quantity"] = -trades["position"]

    trades["pnl_gross"] = (
        trades["position"] * trades["realized_spread_change"]
    )

    return (
        trades
        .sort_values(["model", "feature_set", "date"])
        .reset_index(drop=True)
    )

def run_spread_backtest(
    predictions,
    monthly_prices,
    cost_levels=(0, 1, 5, 10),
    base_quantity=1.0,
    target_vol=0.10,
    max_quantity_multiple=2.0,
):
    """Backtest fixed and risk-scaled equal-barrel spread positions."""

    # Add an unconditional long-spread benchmark
    constant_spread = (
        predictions.drop_duplicates("date")
        .assign(
            model="constant_spread",
            feature_set="benchmark",
            y_pred=1.0,
        )
    )

    signals = pd.concat(
        [predictions, constant_spread],
        ignore_index=True,
    )

    keys = ["model", "feature_set"]

    if signals.duplicated(keys + ["date"]).any():
        raise ValueError("Duplicate forecasts within a strategy.")

    trades = build_spread_positions(signals, monthly_prices)

    # Historical volatility available at each decision date
    risk = (
        monthly_prices[["date", "Brent", "WTI"]]
        .sort_values("date")
        .copy()
    )
    risk["spread_vol"] = (
        (risk["Brent"] - risk["WTI"])
        .diff()
        .rolling(12)
        .std()
    )

    trades = (
        trades.merge(
            risk[["date", "spread_vol"]],
            on="date",
            how="left",
            validate="many_to_one",
        )
        .sort_values(keys + ["date"])
        .reset_index(drop=True)
    )

    if trades["spread_vol"].isna().any():
        raise ValueError("Missing historical volatility estimates.")

    # One common reference capital for every strategy
    first_date = trades["date"].min()
    first = trades.loc[trades["date"].eq(first_date)].iloc[0]

    capital = base_quantity * (
        abs(first["Brent"]) + abs(first["WTI"])
    )

    if capital <= 0:
        raise ValueError("Reference capital must be positive.")

    quantity_cap = base_quantity * max_quantity_multiple

    risk_quantity = np.divide(
        target_vol * capital,
        trades["spread_vol"].to_numpy() * np.sqrt(12),
        out=np.full(len(trades), quantity_cap, dtype=float),
        where=trades["spread_vol"].to_numpy() > 0,
    )
    risk_quantity = np.minimum(risk_quantity, quantity_cap)

    results = []

    for sizing, unsigned_quantity in [
        ("fixed", base_quantity),
        ("risk_scaled", risk_quantity),
    ]:
        strategy = trades.copy()
        strategy["sizing"] = sizing

        strategy["quantity"] = (
            np.sign(strategy["y_pred"]) * unsigned_quantity
        )
        strategy["brent_quantity"] = strategy["quantity"]
        strategy["wti_quantity"] = -strategy["quantity"]

        previous_quantity = (
            strategy.groupby(keys)["quantity"]
            .shift(1)
            .fillna(0)
        )

        entry_notional = (
            (strategy["quantity"] - previous_quantity).abs()
            * (strategy["Brent"].abs() + strategy["WTI"].abs())
        )

        # Liquidate the final holding at the final exit prices
        is_last = ~strategy.duplicated(keys, keep="last")

        exit_notional = np.where(
            is_last,
            strategy["quantity"].abs()
            * (
                strategy["next_Brent"].abs()
                + strategy["next_WTI"].abs()
            ),
            0.0,
        )

        strategy["traded_notional"] = entry_notional + exit_notional
        strategy["reference_capital"] = capital

        strategy["pnl_gross"] = (
            strategy["quantity"]
            * strategy["realized_spread_change"]
        )

        for cost_bps in cost_levels:
            scenario = strategy.copy()
            scenario["cost_bps"] = cost_bps

            scenario["transaction_cost"] = (
                scenario["traded_notional"] * cost_bps / 10_000
            )
            scenario["pnl_net"] = (
                scenario["pnl_gross"] - scenario["transaction_cost"]
            )

            results.append(scenario)

    return pd.concat(results, ignore_index=True)

def summarize_spread_backtest(backtest):
    """Summarize P&L normalized by fixed initial reference capital."""
    rows = []

    group_columns = [
        "model", "feature_set", "sizing", "cost_bps",
    ]

    for configuration, group in backtest.groupby(group_columns):
        group = group.sort_values("date")
        capital = group["reference_capital"].iloc[0]

        normalized_pnl = group["pnl_net"] / capital
        monthly_vol = normalized_pnl.std(ddof=1)

        equity = capital + group["pnl_net"].cumsum()
        running_peak = equity.cummax().clip(lower=capital)
        drawdown = equity / running_peak - 1

        traded_notional = group["traded_notional"].sum()
        gross_pnl = group["pnl_gross"].sum()

        rows.append({
            **dict(zip(group_columns, configuration)),
            "months": len(group),
            "gross_return": gross_pnl / capital,
            "net_return": group["pnl_net"].sum() / capital,
            "annualized_vol": monthly_vol * np.sqrt(12),
            "sharpe": (
                normalized_pnl.mean() / monthly_vol * np.sqrt(12)
                if monthly_vol > 0 else np.nan
            ),
            "max_drawdown": -drawdown.min(),
            "annualized_turnover": (
                traded_notional / capital / len(group) * 12
            ),
            "break_even_bps": (
                gross_pnl / traded_notional * 10_000
                if traded_notional > 0 else np.nan
            ),
            "capital_breach": equity.le(0).any(),
        })

    return pd.DataFrame(rows)