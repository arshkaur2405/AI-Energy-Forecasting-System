import pandas as pd

def load_and_clean_data(file_path):
    """
    Loads the PJM Hourly Energy dataset and prepares it for analysis.
    """

    # Load dataset (assuming first column is datetime)
    df = pd.read_csv(file_path, parse_dates=[0], index_col=0)

    # Sort index (VERY important for time series)
    df = df.sort_index()

    # Handle missing values using interpolation
    if df.isnull().values.any():
        df = df.interpolate(method='linear')

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    return df