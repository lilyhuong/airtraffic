import streamlit as st
import pandas as pd 

import streamlit as st
from PIL import Image


st.title('Traffic Forecast')
Home_airport = ('LGW', 'LIS')
Paired_airport = ('FUE', 'AMS')
# df = pd.read
with st.sidebar:
    home_airport = st.selectbox(
        'Home Airport', Home_airport
    )

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

# Load the image from file
image = Image.open("/Users/lilyhuong/Desktop/Amse mag3/semestre 2/Forecast air traffic/airtraffic/images/LGW_AMS.png")

# Display the image using Streamlit
st.image(image, caption='Image Caption', use_column_width=True)

# st.write(df.querry('home_airport = "{}"'.format(home_airport)).shape[0])
