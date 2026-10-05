# Project: Forecasting the Brent–WTI Spread Using Oil Market Fundamentals

## Objective

The objective of this project is to investigate whether oil market fundamentals improve the prediction of the **Brent–WTI spread** compared with models based solely on historical price data. In a second step, these forecasts will be used to build and evaluate different portfolio strategies.

---

## Installation

```bash
git clone <repository-url>
cd <repository-name>
./setup_env.sh
```

---

# 1. Data

The project relies on two main categories of data.

### Oil Prices (EIA / FRED)

We downloaded historical spot price series for:

- Brent (Europe)
- WTI (United States)

These data are used to compute:

- the Brent–WTI spread;
- returns;
- volatility;
- correlations.

### Oil Market Fundamentals (JODI Oil and EIA)

Fundamental data mainly come from **JODI Oil** (monthly frequency), complemented when necessary with **EIA** data for the United States.

The main variables collected are:

- Oil inventories;
- Inventory changes;
- Production;
- Imports;
- Exports;
- Refinery intake.

The dataset covers the **United States** and several **European countries**, with observations starting in the early 2000s.

---

# 2. Analysis and Forecasting

This section focuses on data preparation, feature engineering, and the comparison of several forecasting models for the Brent–WTI spread.

The objective is to assess whether incorporating oil market fundamentals improves predictive performance compared with models relying only on historical prices. Several regression and machine learning approaches will be evaluated using an out-of-sample validation framework.

---

# 3. Portfolio Construction

The forecasts obtained in the previous step are then used to build different investment strategies based on the Brent–WTI spread.

The strategies will be evaluated both before and after transaction costs and compared with simple benchmark strategies to determine whether the predictive models generate economic value.