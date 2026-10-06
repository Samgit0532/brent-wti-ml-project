import matplotlib.pyplot as plt

def plot_spot_prices(prices):
    """
    Plot Brent and WTI spot prices on the same figure.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned dataframe containing the columns:
        - date
        - Brent
        - WTI

    Returns
    -------
    None
        Displays a line plot of the two spot price series.
    """

    plt.figure(figsize=(12, 6))

    plt.plot(prices["date"], prices["Brent"], label="Brent")
    plt.plot(prices["date"], prices["WTI"], label="WTI")

    plt.title("Brent and WTI Spot Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD/barrel)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

def plot_spread(prices):
    """
    Compute and plot the Brent-WTI spread.

    The spread is defined as:
        Brent price - WTI price

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned dataframe containing the columns:
        - date
        - Brent
        - WTI

    Returns
    -------
    None
        Displays a line plot of the Brent-WTI spread.
    """

    spread = prices["Brent"] - prices["WTI"]

    plt.figure(figsize=(12, 6))

    plt.plot(prices["date"], spread, label="Brent - WTI")

    plt.title("Brent-WTI Spread")
    plt.xlabel("Date")
    plt.ylabel("Spread (USD/barrel)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


