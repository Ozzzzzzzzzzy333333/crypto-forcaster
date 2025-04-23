# imports
import streamlit as st

st.set_page_config(page_title="Help", layout="centered")

# CSS
st.markdown("""
    <style>
    .label {
        text-align: center;
        margin-top: 0.5rem;
        font-size: 0.5rem;
        color: #888;
    }
    /* Center the title and subtitle */
    .centered-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="centered-text">Help</h1>', unsafe_allow_html=True)
st.markdown("""
<h3 class="centered-text">
Guide:
<br>
<br>
This app allows for easy to use machine learning predictions for cryptocurrencys.
<br>
Step one select the cryptocurrency you want to predict.
<br>
Step two select the time frame you want to predict for.
<br>
Step three select the model you want to use.
<br>
Finally click the predict button and wait for the model to run.
<br>
And thats it! The model will run and give you a prediction which will be displayed on the main price chart.
<br>
You can also see the prediction history and the model training details by clicking the button called info.
<br>
<br>
Indicators & Charts:
<br>
<br>
The app uses a variety of indicators and charts to help with the prediction process.
<br>
The main chart shows the price movement over time.
<br>
you can view a variety of indicators such as the RSI, MACD, and Bollinger Bands.
<br>
these will be displayed on the main chart or in a separate chart bellow.
<br>
Below the charts is a simple to understand summary of the indicators,
<br>
indicating whether the indicator is bullish or bearish.
<br>
(more information on the indicators can be found in the info section)
<br>
<br>
Common issues and solutions:
<br>
<br>
If the models are not running or the predictions are not showing, first reload the page,
<br>
if that doesnt resolve the issue, restarting the application should fix any issues.
<br>
Lagging data, a small amount of lagging data is expected, this is due to the nature of the 
<br>
models and the data being used. However, if the data is lagging too much, simply reloading 
<br>
the page should fix the issue.
<br>
If you have any other issues, please contact the developer.
</h3>
""", unsafe_allow_html=True)
