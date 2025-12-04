"""
One-time script to convert the raw options data from CSV to Parquet format.

Parquet is a columnar storage format that is highly efficient to read and
stores data types, which eliminates pandas DtypeWarnings and speeds up
application startup significantly.

To run this script:
1. Make sure you have pyarrow installed:
   pip install pyarrow
2. Run from your terminal:
   python convert_to_parquet.py
"""

import pandas as pd
import os


def convert_csv_to_parquet(
    csv_path: str = '../data/spy_options_2020_2022.csv',
    parquet_path: str = '../data/spy_options_2020_2022.parquet'
):
    """
    Loads the raw CSV, cleans it, and saves it as a Parquet file.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Source CSV file not found at '{csv_path}'.")
        return

    print(f"Loading raw CSV from '{csv_path}'...")

    # Define dtypes to handle mixed types during initial load
    dtypes = {
        '[QUOTE_UNIXTIME]': 'int64', '[QUOTE_READTIME]': 'object',
        '[QUOTE_DATE]': 'object', '[QUOTE_TIME_HOURS]': 'float64',
        '[UNDERLYING_LAST]': 'float64', '[EXPIRE_DATE]': 'object',
        '[EXPIRE_UNIX]': 'int64', '[DTE]': 'float64',
        '[C_DELTA]': 'float64', '[C_GAMMA]': 'float64',
        '[C_VEGA]': 'float64', '[C_THETA]': 'float64',
        '[C_RHO]': 'float64', '[C_IV]': 'float64',
        '[C_VOLUME]': 'float64', '[C_LAST]': 'float64',
        '[C_SIZE]': 'object', '[C_BID]': 'float64',
        '[C_ASK]': 'float64', '[STRIKE]': 'float64',
        '[P_BID]': 'float64', '[P_ASK]': 'float64',
        '[P_SIZE]': 'object', '[P_LAST]': 'float64',
        '[P_DELTA]': 'float64', '[P_GAMMA]': 'float64',
        '[P_VEGA]': 'float64', '[P_THETA]': 'float64',
        '[P_RHO]': 'float64', '[P_IV]': 'float64',
        '[P_VOLUME]': 'float64',
    }

    df = pd.read_csv(csv_path, dtype=dtypes, na_values=[' '])

    print("Cleaning data and converting types...")
    # Clean column names: remove brackets and leading/trailing whitespace
    df.columns = df.columns.str.strip().str.replace(r'\[|\]', '', regex=True)

    # Convert date columns to datetime objects
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])
    df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'])

    # Ensure data is sorted
    df.sort_values(by=['QUOTE_DATE', 'STRIKE'], inplace=True)

    try:
        print(f"Saving cleaned data to Parquet file: '{parquet_path}'...")
        df.to_parquet(parquet_path, engine='pyarrow')
        print("\nConversion successful!")
        print(f"You can now update 'engine/replay.py' to use '{os.path.basename(parquet_path)}'.")
    except ImportError:
        print("\nError: 'pyarrow' is not installed. Please install it to save to Parquet format.")
        print("Install using: pip install pyarrow")


if __name__ == "__main__":
    convert_csv_to_parquet()