"""
Market Data Replay Engine.

This module provides a class to simulate a live feed of options market data
from a historical CSV file. It reads a comprehensive dataset and yields
data day-by-day for a specific, chosen expiration.
"""

import time
from typing import Dict, Any, Iterator

import numpy as np
import pandas as pd


class MarketReplay:
    """
    Simulates a live market feed from a historical options data CSV.

    This class loads a dataset of SPY options from 2020-2022 and provides
    a generator method to stream the data one trading day at a time,
    focusing on a single expiration date closest to a target DTE (Days to
    Expiration).

    Attributes:
        df (pd.DataFrame): The loaded and pre-processed options data.
        unique_dates (pd.DatetimeIndex): Sorted unique quote dates from the dataset.
    """

    def __init__(self, data_path: str = 'data/spy_options_2020_2022.parquet'):
        """
        Initializes the MarketReplay engine by loading and processing the data.

        It will try to load the efficient Parquet file first, and fall back
        to the original CSV if the Parquet file is not found.

        Args:
            data_path: The file path to the options data (Parquet or CSV).
        """
        try:
            if data_path.endswith('.parquet'):
                self.df = pd.read_parquet(data_path)
                print("Successfully loaded fast Parquet data.")
            elif data_path.endswith('.csv'):
                # This fallback logic is kept just in case, but is not recommended
                print("Warning: Loading from slow CSV. Run convert_to_parquet.py for better performance.")
                # (The original CSV loading logic would go here if needed)
                raise NotImplementedError("CSV loading is deprecated. Please convert to Parquet.")
            else:
                raise ValueError("Unsupported data file format. Please use .parquet.")

        except FileNotFoundError:
            print(f"Error: The data file was not found at '{data_path}'.")
            print("Please ensure the CSV file is in the correct directory.")
            # Create an empty dataframe to prevent crashes on subsequent calls
            self.df = pd.DataFrame()

        if not self.df.empty:
            # Sort by date and strike to ensure data is in a predictable order
            self.df.sort_values(by=['QUOTE_DATE', 'STRIKE'], inplace=True)
            self.unique_dates = self.df['QUOTE_DATE'].unique()

    def stream_day_by_day(self, speed_seconds: float = 1.0) -> Iterator[Dict[str, Any]]:
        """
        A generator that yields market data for each unique trading day.

        For each day, it identifies the single expiration date closest to 45
        days out and yields the relevant market data for that chain.

        Args:
            speed_seconds: The delay in seconds between yielding each day's
                           data to simulate a real-time feed.

        Yields:
            A dictionary containing the market state for a single day and
            a single expiration, with the format:
            {'date': pd.Timestamp, 'spot': float, 'T': float,
             'strikes': np.ndarray, 'mid_ivs': np.ndarray}
        """
        if self.df.empty:
            # If the dataframe is empty (e.g., file not found), do nothing.
            return

        for date in self.unique_dates:
            day_df = self.df[self.df['QUOTE_DATE'] == date]
            if day_df.empty:
                continue

            # Find the expiration date (DTE) closest to 45 days
            target_dte = 45
            closest_dte = day_df.iloc[(day_df['DTE'] - target_dte).abs().argmin()]['DTE']

            # Filter to get the single, chosen expiration chain
            chain_df = day_df[day_df['DTE'] == closest_dte].copy()

            # Clean data: drop zero bids, ensure IV is reasonable
            chain_df = chain_df[(chain_df['C_BID'] > 0) & (chain_df['P_BID'] > 0)]
            chain_df = chain_df[(chain_df['C_IV'] > 0.01) & (chain_df['P_IV'] > 0.01)]

            if chain_df.empty:
                continue

            # Calculate the mid-point implied volatility
            chain_df['MID_IV'] = (chain_df['C_IV'] + chain_df['P_IV']) / 2.0

            # Extract the necessary data for the payload
            spot_price = chain_df['UNDERLYING_LAST'].iloc[0]
            time_to_expiry = chain_df['DTE'].iloc[0] / 365.0

            # Yield the structured data for the current day
            yield {
                'date': date,
                'spot': spot_price,
                'T': time_to_expiry,
                'strikes': chain_df['STRIKE'].values,
                'mid_ivs': chain_df['MID_IV'].values,
            }

            # Simulate delay for the "live" feed
            time.sleep(speed_seconds)