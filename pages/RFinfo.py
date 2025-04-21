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