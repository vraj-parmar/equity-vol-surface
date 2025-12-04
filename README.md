# Equity Volatility Surface Dashboard

This project is a high-performance Streamlit dashboard that provides a "control room" for visualising the equity volatility surface. It replays historical options data for SPY (2020-2022), calibrates a smooth volatility curve in real-time, and calculates live risk metrics for a sample portfolio.

The application is designed to be modular, performant, and visually intuitive, showcasing a complete workflow from raw data ingestion to advanced financial modelling and visualisation.

![Dashboard Screenshot](./docs/dashboard_animation.gif)
---

## Features

- **Real-Time Data Replay**: A "time machine" engine reads historical options data and streams it day-by-day as if it were a live feed.
- **Advanced Volatility Modelling**: Implements the **SVI (Stochastic Volatility Inspired)** model to fit a smooth, arbitrage-free curve to noisy market implied volatilities.
- **Interactive & Animated Charting**: Uses Plotly to create a smooth, animated, dual-axis chart that updates in real-time, showing:
  - Market Implied Volatility (scatter points)
  - Fitted SVI Volatility Curve (line)
  - 25-Delta Skew (horizontal line on a secondary axis)
- **Live Risk Panel**: Calculates and displays the **Portfolio Delta** for a hypothetical position (10 ATM straddles) and shows a visual alert if risk limits are breached.
- **High-Performance Core**:
  - The Black-Scholes and Greeks calculations are accelerated with `numba`.
  - Data is loaded from the highly efficient **Parquet** file format to ensure fast application startup and no data-type warnings.
- **Modular Architecture**: The code is cleanly separated into a `core` (math), `engine` (data), and `app.py` (UI).

---

## Key Learnings & Concepts

This project serves as a practical application of several key concepts in quantitative finance and software engineering:

1.  **Financial Engineering & Modelling**:
    *   **Option Pricing**: Implemented the core Black-Scholes-Merton model for European options.
    *   **Risk Management**: Calculated the "Greeks" (Delta, Gamma, Vega, etc.) to understand and quantify portfolio risk.
    *   **Implied Volatility**: Used the Newton-Raphson numerical method to back-solve for the market's expectation of volatility.
    *   **Volatility Surface Parametrisation**: Modelled the volatility smile using the **SVI (Stochastic Volatility Inspired)** model, converting noisy market data into a smooth, arbitrage-free curve.

2.  **Python for High-Performance Computing**:
    *   **Code Acceleration**: Applied `numba`'s Just-In-Time (JIT) compiler to critical, loop-heavy mathematical functions, resulting in C-like speed for Python code.
    *   **Efficient Data Handling**: Leveraged `pandas` for data manipulation and demonstrated the performance benefits of switching from CSV to the **Parquet** columnar storage format, which eliminates data type ambiguity and drastically speeds up I/O.
    *   **Scientific Stack**: Utilised `numpy` for vectorised numerical operations and `scipy` for its optimisation library (`minimize`) to calibrate the SVI model.

3.  **Software & Application Architecture**:
    *   **Modular Design**: Structured the project into a clean, multi-layered architecture (`core`, `engine`, `UI`) for maintainability and scalability.
    *   **Interactive Data Visualisation**: Built a responsive user interface with `streamlit` and created smooth, animated charts with `plotly` by updating figure data in-place rather than re-rendering.
    *   **Data Simulation**: Developed a replay engine that acts as a "time machine," a common pattern used in backtesting trading strategies and simulating market environments.
---

## Project Structure

```
.
├── app.py                      # Main Streamlit dashboard application
├── core/
│   ├── bs_pricing.py           # Black-Scholes-Merton pricing and Greeks
│   └── svi_calibration.py      # SVI model calibration engine
├── data/
│   ├── spy_options_2020_2022.csv     # (Source) Raw historical options data
│   └── spy_options_2020_2022.parquet # (Used by app) Efficient data file
├── engine/
│   ├── replay.py               # The "time machine" data streaming engine
│   └── convert_to_parquet.py   # One-time script to convert CSV to Parquet
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

---

## Setup and Installation

Follow these steps to get the dashboard running on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

### 3. Install Dependencies

First, ensure you have the `requirements.txt` file below in your root directory. Then, run:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Data

The application is designed to run using the efficient Parquet file format.

- Place your raw `spy_options_2020_2022.csv` file inside the `data/` directory.
- Run the one-time conversion script. This will read the CSV, clean it, and save it as `spy_options_2020_2022.parquet`.

```bash
python engine/convert_to_parquet.py
```

### 5. Run the Streamlit App

You are now ready to launch the dashboard!

```bash
streamlit run app.py
```

Open your browser to the local URL provided by Streamlit (usually `http://localhost:8501`). Use the "Start/Stop Replay" button in the sidebar to begin the simulation.

---

## Dependencies

The project's dependencies are listed in `requirements.txt`.

```
streamlit
pandas
numpy
scipy
plotly
numba
pyarrow
```