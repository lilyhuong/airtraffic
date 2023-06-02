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
import streamlit as st
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from mlforecast import MLForecast
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean
 
        
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

def predict_prophet(home_airport, paire_airport, forecast_day, nb):
    
    traffic_df = pd.read_parquet("/Users/lilyhuong/Desktop/Amse mag3/semestre 2/Forecast air traffic/airtraffic/traffic_10lines.parquet")
    df1 = generate_route_df(traffic_df, home_airport, paire_airport)
    
    nextday = forecast_day + timedelta(days = nb)
    if nextday > df1["date"].iloc[-1]:
        nb_forecast = (nextday - df1["date"].iloc[-1].to_pydatetime().date()).days
    else:
        nb_forecast = 15
        
        
    #predict model
    _filename = '/Users/lilyhuong/Desktop/Amse mag3/semestre 2/Forecast air traffic/airtraffic/notebooks/route_model_prophet_{home}_{paired}.json'.format(home = home_airport, paired = paire_airport)
    a = load(_filename)
        
    #prepare prediction for next X days 
    future_df = a.make_future_dataframe(periods= nb_forecast) 
    result = a.predict(future_df)
    res = result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    res = res[(res.ds >= pd.to_datetime(forecast_day))][(res.ds <= pd.to_datetime(nextday))]
    res = res.rename(columns={"ds": "date", "yhat": "predicted value", "yhat_lower":"lower bound", "yhat_upper":"upper bound"})
    
    result = pd.merge(res, df1, how="left", on=["date"])[["date", "predicted value", "pax_total"]].rename(columns={"pax_total": "real value"})
    ### plot the result of prediction 
    return result

    
def plot_result(data):
        # Plot the time series
    plt.plot(data['date'], data['predicted value'], label='predicted value')
    plt.plot(data['date'], data['real value'], label='real value')

    # Set plot title and labels
    plt.title('Result of number of passengers by prediction')
    plt.xlabel('Date')
    plt.ylabel('Number of passengers')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.show()
    
    return plt

@njit
def rolling_mean_28(x):
    return rolling_mean(x, window_size=28)

def predict_Nixtla(home_airport, paire_airport, forecast_day, nb):
    
    traffic_df = pd.read_parquet("/Users/lilyhuong/Desktop/Amse mag3/semestre 2/Forecast air traffic/airtraffic/traffic_10lines.parquet")
    df1 = generate_route_df(traffic_df, home_airport, paire_airport)
    
    # nextday = forecast_day + timedelta(days = nb)
    # if nextday > df1["date"].iloc[-1]:
    #     nb_forecast = (nextday - df1["date"].iloc[-1].to_pydatetime().date()).days
    # else:
    #     nb_forecast = 15
           
    # parametre le modele    
    models = [
    lgb.LGBMRegressor(),
    xgb.XGBRegressor(),
    RandomForestRegressor(random_state=0),
    ]
    
    fcst = MLForecast(
        models=models,
        freq='D',
        lags=[7, 14, 21, 28],
        lag_transforms={
            1: [expanding_mean],
            7: [rolling_mean_28]
        },
        date_features=['dayofweek'],
        differences=[1],
    )
    #selectionner seulement le dataframe avec la date < la date selectionner pour faire la prédiction 
    df1 = df1[df1.date <= pd.to_datetime(forecast_day)]
    nixtla_model = fcst.fit(df1.drop(columns = ['paired_airport']),id_col = 'home_airport', time_col= 'date', target_col= 'pax_total')
    predict_df = (nixtla_model.predict(nb)).drop(columns = ['home_airport'])
    predict_df