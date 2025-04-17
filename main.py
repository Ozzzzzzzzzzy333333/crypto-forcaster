import streamlit as st

# Page Config
st.set_page_config(page_title="Market Selector", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 5rem;
        margin-top: 5rem;
    }
    .circle-button {
        height: 150px;
        width: 150px;
        border-radius: 50%;
        background-color: #111;
        color: white;
        font-size: 4rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 0px rgba(0,0,0,0);
        transition: all 0.3s ease-in-out;
        cursor: pointer;
        text-decoration: none;
    }
    .crypto:hover {
        box-shadow: 0 0 30px orange;
        background-color: #222;
    }
    .stocks:hover {
        box-shadow: 0 0 30px green;
        background-color: #222;
    }
    .forex:hover {
        box-shadow: 0 0 30px yellow; 
        background-color: #222;
    }
    .label {
        text-align: center;
        margin-top: 0.5rem;
        font-size: 1.2rem;
        color: #888;
    }
    /* Hide Streamlit sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title(" Market Prediction App")
st.subheader("Choose your market to begin")

# Buttons (styled as links)
st.markdown("""
<div class="container">
    <a href="/Crypto" class="circle-button crypto">₿</a>
    <a href="/Stocks" class="circle-button stocks">🗠</a>
    <a href="/Forex" class="circle-button forex">£</a>
</div>
<div class="container">
    <div class="label">Crypto</div>
    <div class="label">Stocks</div>
    <div class="label">Forex</div>
</div>
""", unsafe_allow_html=True)

