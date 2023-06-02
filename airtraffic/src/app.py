import streamlit as st
import pandas as pd 

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

from predict_model import (generate_route_df, predict)


st.title('Traffic Forecast')
Home_airport = ('LGW', 'LIS', "LYS", "NTE", "PNH", "POP", "SCL", "SSA")
#Paired_airport = ('FUE', 'AMS')

# Define the options for the second select box based on the selected value of the first select box
# df = pd.read
with st.sidebar:
    home_airport = st.selectbox(
        'Home Airport', Home_airport
    )
    # Define the options for the second select box based on the selected value of the first select box
    if home_airport == 'LGW':
        Paired_airport = ("AMS", "BCN")
    elif home_airport  == 'LIS':
        Paired_airport = ("OPO", "ORY")
    elif home_airport  == 'LYS':
        Paired_airport = ("PIS",)
    elif home_airport  == 'NTE':
        Paired_airport = ("FUE",)  
    elif home_airport  == 'PNH':
        Paired_airport = ("NGB",) 
    elif home_airport  == 'POP':
        Paired_airport = ("JFK",)  
    elif home_airport  == 'SCL':
        Paired_airport = ("LHR",) 
    elif home_airport  == 'SSA':
        Paired_airport = ("GRU",) 
    
    paired_airport = st.selectbox(
        'Paired Airport', Paired_airport
    )
    
    
    forecast_date = st.date_input('Forecast Start Date')
    nb_days = st.slider("Day of forecats", 7, 30, 1)
    run_forecast = st.button("Forecast")

st.write('Home airport select', home_airport)
st.write('paired airport select', paired_airport)
st.write('Day select', nb_days)
st.write('Date select', forecast_date)

# Display the image using Streamlit
link_image = "/Users/lilyhuong/Desktop/Amse mag3/semestre 2/Forecast air traffic/airtraffic/images/"
file_img = link_image + home_airport + "_" + paired_airport + ".png"
st.image(file_img)

#display forecast 
if run_forecast:
    table = predict(home_airport, paired_airport, forecast_date, nb_days)
    st.dataframe(table)


# st.dataframe(table)

# st.write(df.querry('home_airport = "{}"'.format(home_airport)).shape[0])
