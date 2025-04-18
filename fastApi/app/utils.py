import pandas as pd

def preprocess_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.date
    df["weekday"] = df["timestamp"].dt.day_name()
    return df

