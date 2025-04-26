# imports
import json
import streamlit as st

st.set_page_config(page_title="LSTM Info", layout="centered")

# CSS
st.markdown("""
    <style>
    .label {
        text-align: center;
        margin-top: 0.5rem;
        font-size: 0.5rem;
        color: #888;
    }
    /* Center the title and subtitale */
    .centered-text {
        text-align: center;
    }
    
    </style>
""", unsafe_allow_html=True)

# read LSTM log file
def read_lstm_log():
    try:
        with open('lstm_log.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"lstm_trained": False}

# Read the LSTM log
lstm_log = read_lstm_log()

st.markdown('<h1 class="centered-text">LSTM Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<h3 class="centered-text">
Long Short-Term Memory (LSTM) is a more complex machine learning technique than Random Forest algorithms. This network is a type of recurrent neural network (RNN)
that is capable of learning long-term dependencies.

LSTM is a type of neural network designed to work with sequences of data over time and therefore, it is perfect for stock or crypto prices. It remembers past prices and uses that memory to guess what’s coming next,
because of this, it’s especially good at spotting patterns that unfold over time.

It takes in time-series data  and then it processes the sequence one step at a time,
remembering important details and forgetting irrelevant ones.

LSTM are part of the RNN family (Recurrent Neural Networks), but they are better at learning
long-term patterns, meaning that it can better see the bigger picture.

whilst it is typically more accurate than Random Forests, it is also more complex to build and train.
It also requires significant data and computing power and is harder to interpret than Random Forests.
Due to this, we recommend using the RF as a rough guide for singular predictions and the LSTM for more
complex patterns or wider timeframes.

</h3>
""", unsafe_allow_html=True)
if not lstm_log.get("lstm_trained", False):
    st.markdown('<h3 class="centered-text">Sorry, the model hasn\'t been generated and run yet.</h3>', unsafe_allow_html=True)
else:
    # technical details
    st.markdown('<h1 class="centered-text">Technical Details</h1>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <h3 class="centered-text">
    <b>Training Details:</b><br>
    Features Used: {', '.join(lstm_log['features'])}<br>
    Cryptocurrency: {lstm_log['crypto']}<br>
    Time Interval: {lstm_log['interval']}<br>
    Current Price: {lstm_log['current_price']}<br>
    </h3>
    """, unsafe_allow_html=True)
    
    # display predictions
    st.markdown('<h1 class="centered-text">Predictions</h1>', unsafe_allow_html=True)
    for pred in lstm_log.get("predictions", []):
        st.markdown(f"""
        <h3 class="centered-text">
        Prediction Start Time: {pred['prediction_start_time']}<br>
        Prediction End Time: {pred['prediction_end_time']}<br>
        Movement: {pred['movement']}<br>
        Current Price: {pred['current_price']}<br>
        Predicted Price: {pred['predicted_price']}<br>
        Confidence: {pred['confidence']:.2f}<br>
        Turning Points: {pred.get('turning_points', 'N/A')}<br>
        </h3>
        """, unsafe_allow_html=True)