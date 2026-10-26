# Brent–WTI Spread Forecasting

A machine-learning project comparing price-based forecasts with oil market fundamentals, followed by a long–short trading backtest.

The economic motivation, methodology, results, and references are presented in `main.ipynb`.

## Installation

Requirements: Git and Conda (Anaconda or Miniconda).

```bash
git clone https://github.com/Samgit0532/brent-wti-ml-project.git
cd brent-wti-ml-project
bash setup_env.sh
```

The setup script creates or updates the `brent_wti_project` environment and registers the Jupyter kernel **Python (Brent-WTI)**.

## Run the project

From the project root:

```bash
conda activate brent_wti_project
jupyter lab
```

Open `main.ipynb`, select **Python (Brent-WTI)**, and run the cells in order using **Restart Kernel and Run All Cells**.

Alternatively, open the repository in VS Code and select the same notebook kernel.

Run the notebook from the project root: data paths and imports from `scripts/` depend on this working directory.

## Required data

The installation script installs dependencies; it does not download the datasets.

The notebook expects:

| Location | Contents |
|---|---|
| `data/EIA/spot_brent.csv` | Daily Brent spot prices exported from FRED |
| `data/EIA/spot_wti.csv` | Daily WTI spot prices exported from FRED |
| `data/JODI/` | Yearly JODI Oil CSV files |

The FRED files should contain a date column and one price column.

The JODI files should include:
`REF_AREA`, `TIME_PERIOD`, `ENERGY_PRODUCT`, `FLOW_BREAKDOWN`,
`UNIT_MEASURE`, `OBS_VALUE`, and `ASSESSMENT_CODE`.

Sources:

- [Brent prices — FRED](https://fred.stlouisfed.org/series/DCOILBRENTEU)
- [WTI prices — FRED](https://fred.stlouisfed.org/series/DCOILWTICO)
- [JODI Oil](https://www.jodidata.org/oil/)

## What the notebook runs

1. **Data preparation:** clean daily prices, construct monthly observations, and aggregate JODI fundamentals.
2. **Predictor construction:** build price variables, transform fundamentals, and define five information sets.
3. **Forecasting:** generate rolling forecasts using statistical benchmarks, Ridge, Random Forest, and LightGBM.
4. **Evaluation:** compare forecast errors, directional accuracy, stability over time, and model diagnostics.
5. **Trading:** convert forecasts into equal-barrel Brent–WTI positions and compare fixed and risk-scaled quantities before and after transaction costs.

Tables and figures are displayed directly in the notebook. Rolling model fitting is the most computationally intensive step.

## Code organization

| File | Role |
|---|---|
| `main.ipynb` | Research workflow, results, and interpretation |
| `scripts/cleaning.py` | Price and JODI cleaning |
| `scripts/features.py` | Monthly predictors, aggregates, transformations, and merging |
| `scripts/models.py` | Model definitions, rolling forecasts, and forecast metrics |
| `scripts/backtest.py` | Positions, transaction costs, and strategy metrics |
| `scripts/plots.py` | Reusable plotting functions |
| `scripts/tests.py` | Statistical diagnostic helpers |
| `environment.yml` | Conda dependencies |
| `setup_env.sh` | Environment and notebook-kernel setup |

## Configuration and reproducibility

Evaluation dates, training-window length, country groups, and trading assumptions are defined in the notebook. Model settings are defined in `get_ml_models()` in `scripts/models.py`.

After changing a script, restart the kernel and rerun the notebook to avoid using an older imported function.

Results depend on the input data and settings. Updating source files can change country coverage, the available sample, and reported performance.

The trading exercise uses spot prices and simplified execution costs. It is a research simulation rather than an executable futures strategy.