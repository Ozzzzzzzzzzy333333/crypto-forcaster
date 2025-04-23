# imports
import json
import streamlit as st
st.set_page_config(page_title="RF Info", layout="centered")

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

# read rf log
def read_rf_log():
    try:
        with open('rf_log.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"rf_trained": False}

#read the RF log
rf_log = read_rf_log()

st.markdown('<h1 class="centered-text">Random Forest Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<h3 class="centered-text">
A Random Forest is one of the simplest machine learning techniques; however, this
does not mean it can't be highly effective.

It uses a group of decision-makers (called decision trees) that vote on what to do.
Each decision tree in the forest looks at the data and gives its own prediction (in this instance, “price will go up” or “price will go down”), 
and the forest takes a vote, and the most common answer wins.

It builds lots of decision trees using slightly different parts of the trading data.
Each tree is trained to make a prediction, and then their predictions are 
combined to make the final answer. This helps avoid bad guesses by any one tree.
</h3>
""", unsafe_allow_html=True)
if not rf_log.get("rf_trained", False):
    st.markdown('<h3 class="centered-text">Sorry, the model hasn\'t been generated and run yet.</h3>', unsafe_allow_html=True)
else:
    # technical details
    st.markdown('<h1 class="centered-text">Technical Details</h1>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <h3 class="centered-text">
    <b>Training Details:</b><br>
    Cryptocurrency: {rf_log['crypto']}<br>
    Time Interval: {rf_log['interval']}<br>
    Current Price: {rf_log['current_price']}<br>
    </h3>
    """, unsafe_allow_html=True)
    
    # predictions
    st.markdown('<h1 class="centered-text">Predictions</h1>', unsafe_allow_html=True)
    for pred in rf_log.get("predictions", []):
        st.markdown(f"""
        <h3 class="centered-text">
        Prediction Start Time: {pred['prediction_start_time']}<br>
        Prediction End Time: {pred['prediction_end_time']}<br>
        Movement: {pred['movement']}<br>
        Current Price: {pred['current_price']}<br>
        Predicted Price: {pred['predicted_price']}<br>
        </h3>
        """, unsafe_allow_html=True)