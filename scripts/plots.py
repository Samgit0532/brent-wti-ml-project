import matplotlib.pyplot as plt

def plot_prices_and_spread(prices):
    """Plot Brent and WTI prices and their spread."""
    data = prices.sort_values("date")
    spread = data["Brent"] - data["WTI"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(
        data["date"], data["Brent"], label="Brent", linewidth=1
    )
    axes[0].plot(
        data["date"], data["WTI"], label="WTI", linewidth=1
    )
    axes[0].set_title("Brent and WTI spot prices")
    axes[0].set_ylabel("USD / barrel")
    axes[0].legend(frameon=False)

    axes[1].plot(data["date"], spread, color="teal", linewidth=1)
    axes[1].axhline(
        0, color="grey", linewidth=0.8, linestyle="--"
    )
    axes[1].set_title("Brent–WTI spread")
    axes[1].set_ylabel("USD / barrel")
    axes[1].set_xlabel("Date")

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_ridge_penalties(predictions):

    """Display each specification separately on a common log scale."""
    alphas = (
        predictions.loc[predictions["model"].eq("ridge")]
        .pivot(
            index="date",
            columns="feature_set",
            values="ridge_alpha",
        )
        .sort_index()
    )

    labels = {
        "prices_only": "Prices only",
        "prices_wti": "Prices + WTI fundamentals",
        "prices_brent": "Prices + Brent fundamentals",
        "prices_both_regions": "Prices + both regional blocks",
        "prices_pooled": "Prices + pooled fundamentals",
    }

    fig, axes = plt.subplots(
        len(labels),
        1,
        figsize=(11, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for ax, (column, label) in zip(axes[:, 0], labels.items()):
        ax.step(
            alphas.index,
            alphas[column],
            where="post",
            color="teal",
            linewidth=1.3,
        )
        ax.set_yscale("log")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_ylabel("Alpha")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1, 0].set_xlabel("Forecast decision date")
    fig.suptitle("Ridge penalty selected within each training window")

    plt.show()

def plot_forecast_drivers(diagnostics):
    """Compare mean standardized Ridge coefficients and LightGBM gains."""
    summary = (
        diagnostics
        .groupby(["feature", "model"])["value"]
        .mean()
        .unstack("model")
    )

    # Use the same feature order in both panels
    summary = summary.sort_values("lightgbm")

    price_labels = {
        "spread": "Current spread",
        "spread_change": "Latest spread change",
        "spread_change_mean_6": "Six-month trend",
        "spread_change_vol_12": "Twelve-month volatility",
    }

    flow_labels = {
        "CLOSTLV": "inventory growth",
        "STOCKCH": "stock change",
        "INDPROD": "production growth",
        "TOTIMPSB": "import growth",
        "TOTEXPSB": "export growth",
        "REFINOBS": "refinery-intake growth",
    }

    def feature_label(feature):
        if feature in price_labels:
            return price_labels[feature]

        region, flow = feature.removesuffix("_yoy").split("_", 1)
        return f"{region}: {flow_labels.get(flow, flow)}"

    labels = [feature_label(feature) for feature in summary.index]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 8), sharey=True,
        constrained_layout=True,
    )

    colors = [
        "teal" if value >= 0 else "indianred"
        for value in summary["ridge"]
    ]

    axes[0].barh(labels, summary["ridge"], color=colors)
    axes[0].axvline(0, color="grey", linewidth=0.8)
    axes[0].set_title("Ridge — average standardized coefficients")
    axes[0].set_xlabel("USD / barrel per training-sample standard deviation")

    axes[1].barh(labels, summary["lightgbm"], color="steelblue")
    axes[1].set_title("LightGBM — average share of training gain")
    axes[1].set_xlabel("Gain share (%)")

    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    plt.show()   
    
def plot_strategy_performance(backtest, cost_bps=5):
    """Compare cumulative net P&L across models and sizing rules."""
    data = backtest.loc[
        backtest["cost_bps"].eq(cost_bps)
    ].copy()

    models = {
        "ridge": "Ridge",
        "random_forest": "Random Forest",
        "lightgbm": "LightGBM",
    }

    specifications = {
        "prices_only": "Prices only",
        "prices_wti": "Prices + WTI",
        "prices_brent": "Prices + Brent",
        "prices_both_regions": "Prices + both regions",
        "prices_pooled": "Prices + pooled",
    }

    sizings = {
        "fixed": "Fixed quantity",
        "risk_scaled": "Risk-scaled quantity",
    }

    fig, axes = plt.subplots(
        3, 2,
        figsize=(15, 11),
        sharex=True,
        sharey=True,
    )

    def draw_path(ax, group, label, **kwargs):
        group = group.sort_values("realization_date")

        cumulative_pnl = (
            100 * group["pnl_net"].cumsum()
            / group["reference_capital"].iloc[0]
        )

        ax.plot(
            group["realization_date"],
            cumulative_pnl,
            label=label,
            **kwargs,
        )

    for row, (model, model_label) in enumerate(models.items()):
        for col, (sizing, sizing_label) in enumerate(sizings.items()):
            ax = axes[row, col]
            subset = data.loc[data["sizing"].eq(sizing)]

            for specification, label in specifications.items():
                group = subset.loc[
                    subset["model"].eq(model)
                    & subset["feature_set"].eq(specification)
                ]

                draw_path(ax, group, label, linewidth=1.2)

            for benchmark, label, style in [
                ("ar1", "AR(1)", "--"),
                ("constant_spread", "Constant long spread", ":"),
            ]:
                group = subset.loc[
                    subset["model"].eq(benchmark)
                ]

                draw_path(
                    ax, group, label,
                    color="grey",
                    linestyle=style,
                    linewidth=1,
                )

            ax.axhline(0, color="grey", linewidth=0.5)
            ax.set_title(f"{model_label} — {sizing_label}")
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)

            if col == 0:
                ax.set_ylabel("Net P&L / initial capital (%)")

    for ax in axes[-1]:
        ax.set_xlabel("Realization date")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        f"Cumulative net strategy P&L — {cost_bps} bps per traded notional"
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))

    plt.show()     