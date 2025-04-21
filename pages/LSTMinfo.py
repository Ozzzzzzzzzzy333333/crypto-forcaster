import streamlit as st

# Page Config
st.set_page_config(page_title="RF info", layout="centered")

# Custom CSS for styling
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

# Title
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