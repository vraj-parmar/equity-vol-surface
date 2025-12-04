"""
Equity Volatility Surface Dashboard.

This Streamlit application serves as a "Control Room" to visualize the
replay of historical options data, fit a volatility surface in real-time,
and display relevant risk metrics.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import the core components we've built
from engine.replay import MarketReplay
from core.svi_calibration import calibrate_svi, svi_raw
from core.bs_pricing import get_implied_vol, calculate_greeks

# --- Page Configuration ---
st.set_page_config(
    page_title="Equity Volatility Surface Dashboard",
    layout='wide',
    initial_sidebar_state='collapsed'
)

st.title("Equity Volatility Surface: Real-Time Replay & Risk")
st.markdown("This dashboard replays historical SPY options data from 2020-2022, "
            "calibrates an SVI volatility curve, and calculates portfolio risk in real-time.")

# --- Helper Functions ---

def find_iv_for_delta(target_delta: float, option_type: str, S: float, T: float, r: float, q: float, svi_params: dict) -> float:
    """
    Finds the implied volatility for a given delta by searching across a range of strikes.
    This uses the fitted SVI curve to ensure a smooth relationship.
    """
    # Generate a fine-grained range of strikes around the spot price to find the target delta
    search_strikes = np.linspace(S * 0.7, S * 1.3, 300)
    min_delta_diff = float('inf')
    best_strike = S

    for k_strike in search_strikes:
        # Get the smooth IV from our SVI model for this strike
        log_moneyness = np.log(k_strike / S)
        total_variance = svi_raw(log_moneyness, **svi_params)
        if total_variance < 0:
            continue
        sigma = np.sqrt(total_variance / T)

        # Calculate greeks using the SVI vol
        greeks = calculate_greeks(S, k_strike, T, sigma, option_type, r, q)
        delta = greeks['Delta']

        delta_diff = abs(delta - target_delta)
        if delta_diff < min_delta_diff:
            min_delta_diff = delta_diff
            best_strike = k_strike

    # Return the IV for the strike that best matched the target delta
    final_log_moneyness = np.log(best_strike / S)
    final_total_variance = svi_raw(final_log_moneyness, **svi_params)
    return np.sqrt(final_total_variance / T) if final_total_variance > 0 else np.nan


# --- UI Placeholders ---
st.sidebar.header("Controls")
replay_speed = st.sidebar.select_slider(
    "Replay Speed (seconds per day)",
    options=[0.0, 0.25, 0.5, 1.0, 2.0],
    value=0.5
)

# Create empty containers that will be filled and updated in the loop
placeholder_metrics = st.empty()
placeholder_chart = st.empty()
placeholder_risk = st.empty()

# --- Initialization ---
try:
    market_replay = MarketReplay()
    # Initialize session state to control the replay loop
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
except Exception as e:
    st.error(f"Failed to initialize MarketReplay engine: {e}")
    st.stop()


start_button = st.sidebar.button("Start/Stop Replay", type="primary")

if start_button:
    st.session_state.is_running = not st.session_state.is_running

if st.session_state.is_running:
    # --- Main Replay Loop ---
    for day_data in market_replay.stream_day_by_day(speed_seconds=replay_speed):
        if not st.session_state.is_running: # Check if stop was pressed during sleep
            break

        # Unpack data for the current day
        date = pd.to_datetime(day_data['date']).strftime('%Y-%m-%d')
        S = day_data['spot']
        T = day_data['T']
        strikes = day_data['strikes']
        market_ivs = day_data['mid_ivs']

        # --- Core Logic: SVI Calibration ---
        try:
            svi_params, fitted_ivs = calibrate_svi(strikes, market_ivs, T, S)
        except Exception:
            # If calibration fails, skip this day
            continue

        # --- Calculations for UI Components ---
        # 1. ATM IV
        atm_strike_idx = np.abs(strikes - S).argmin()
        atm_iv = market_ivs[atm_strike_idx]

        # 2. Skew
        put_25d_iv = find_iv_for_delta(-0.25, 'put', S, T, 0.04, 0.015, svi_params)
        call_25d_iv = find_iv_for_delta(0.25, 'call', S, T, 0.04, 0.015, svi_params)
        skew = (put_25d_iv - call_25d_iv) * 100 # In vol points

        # 3. Risk Panel: 10 ATM Straddles
        atm_strike = strikes[atm_strike_idx]
        atm_sigma = fitted_ivs[atm_strike_idx]
        
        call_greeks = calculate_greeks(S, atm_strike, T, atm_sigma, 'call')
        put_greeks = calculate_greeks(S, atm_strike, T, atm_sigma, 'put')
        
        # Delta of 1 straddle = Delta Call + Delta Put
        straddle_delta = call_greeks['Delta'] + put_greeks['Delta']
        portfolio_delta = 10 * straddle_delta

        # --- UI Updates ---
        # 1. Top Row Metrics
        with placeholder_metrics.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Date", date)
            col2.metric("SPY Spot Price", f"${S:,.2f}")
            col3.metric("ATM Implied Vol", f"{atm_iv:.2%}")
            col4.metric("25d Skew (Put-Call)", f"{skew:.2f} vol pts")

        # 2. Main Chart (Plotly)
        with placeholder_chart.container():
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Market and Fitted IV
            fig.add_trace(
                go.Scatter(x=strikes, y=market_ivs, mode='markers', name='Market IV', marker=dict(color='blue', opacity=0.6)),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=strikes, y=fitted_ivs, mode='lines', name='SVI Fit IV', line=dict(color='orange', width=3)),
                secondary_y=False,
            )

            # Add a dummy trace for the Skew label
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode='lines', name='Skew', line=dict(color='red', dash='dash')),
                secondary_y=True,
            )
            
            fig.update_layout(
                title_text=f"Volatility Smile for Expiration T = {T*365:.0f} days",
                xaxis_title="Strike Price",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Implied Volatility", secondary_y=False, tickformat=".0%")
            fig.update_yaxes(title_text="Skew", secondary_y=True, showgrid=False, range=[-5, 25], tickvals=[skew], ticktext=[f"{skew:.2f}"])
            
            # Add a horizontal line for the skew value
            fig.add_hline(y=skew, line_width=2, line_dash="dash", line_color="red", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

        # 3. Risk Panel
        with placeholder_risk.container():
            st.subheader("Risk Panel: 10 ATM Straddles")
            col1, col2 = st.columns(2)
            col1.metric("Portfolio Delta", f"{portfolio_delta:+.2f}")

            if abs(portfolio_delta) > 50:
                col2.error("🚨 **Risk Alert:** Delta limit breached!")
            else:
                col2.success("✅ Delta within limits.")

else:
    st.info("Press 'Start/Stop Replay' in the sidebar to begin the simulation.")