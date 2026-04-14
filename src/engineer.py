import pandas as pd
import numpy as np

def build_features(df, target_col):
    """
    Creates temporal, cyclical, and historical features.
    """
    df = df.copy()

    # 1. Temporal Features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear

    # 2. Cyclical Encoding (important for time-series)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    # 3. Lag Features (IMPORTANT FIX)
    for lag in [1, 24, 48, 168]:   # 1 hour, 1 day, 2 days, 1 week
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # 4. Rolling Features
    df['rolling_mean_24'] = df[target_col].shift(1).rolling(window=24).mean()

    return df.dropna()