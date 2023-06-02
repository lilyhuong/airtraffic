import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from joblib import dump, load
from datetime import datetime, timedelta
import logging
#logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
import os

           
        
def generate_route_df(traffic_df: pd.DataFrame, homeAirport:str, paireAirport: str) -> pd.DataFrame:
    

    # """
    # Extract route datafrae from traffic dataframe for route from home airport to paired airport 
    # Args:
    # -traffic_df (pd.DataFrame): traffic datafram
    # - homeAirport (str): IAIA code from airport
    # - pairedAirport (str): IAIA coe for paired airpport
    # Return
    # """

    _df=(traffic_df
        .query ('home_airport == "{}" and paired_airport =="{}"'.format(homeAirport, paireAirport))
        .groupby(['home_airport', 'paired_airport', 'date'])
        .agg(pax_total=('pax', 'sum'))
        .reset_index() 
        )
    return _df

def predict(home_airport, paire_airport, forecast_day, nb):
    # test générer the dataset for NTE and FUE airport 
    traffic_df = pd.read_parquet("/Users/lilyhuong/Desktop/Amse mag3/semestre 2/Forecast air traffic/airtraffic/traffic_10lines.parquet")
    df1 = generate_route_df(traffic_df, home_airport, paire_airport)
    
    nextday = forecast_day + timedelta(days = nb)
    if nextday > df1["date"].iloc[-1]:
        nb_forecast = nextday - df1["date"].iloc[-1].to_pydatetime().date()
    else:
        nb_forecast = 15
        
        
    #predict model
    _filename = '/Users/lilyhuong/Desktop/Amse mag3/semestre 2/Forecast air traffic/airtraffic/notebooks/route_model_prophet_{home}_{paired}.json'.format(home = home_airport, paired = paire_airport)
    with open("_filename", 'r') as f:
        a = model_from_json(f.read())
        
    #prepare prediction for next X days 
    future_df = a.make_future_dataframe(periods= nb_forecast) 
    result = a.predict(future_df)
    res = result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    res = res[(res.ds >= pd.to_datetime(forecast_day))][(res.ds <= pd.to_datetime(nextday))]
    return res 