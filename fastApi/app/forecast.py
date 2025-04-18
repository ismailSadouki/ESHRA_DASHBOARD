from prophet import Prophet
import pandas as pd

def get_forecast(df):
    df_daily = df.groupby('day')['consumption'].sum().reset_index()
    df_daily.columns = ['ds', 'y']

    model = Prophet()
    model.fit(df_daily)

    last_date = df_daily['ds'].max()
    future_dates = pd.date_range(last_date, periods=31, freq='D')[1:]
    future_df = pd.DataFrame({'ds': future_dates})
    forecast = model.predict(future_df)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records")

