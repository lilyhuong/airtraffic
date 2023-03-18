import streamlit as st
import pandas as pd 
st.title('Traffic Forecast')

with st.sidebar:
    home_airport = st.selectbox(
        'Home Airport', ('LGW', 'LIS', 'LYS')
    )
    forecast_date = st.date_input('Forecast Start Date')
    run_forecast = st.button("Forecast")

st.write('Home airport select', home_airport)
st.write('Date select', forecast_date)
