import streamlit as st

# Page Config
st.set_page_config(page_title="RF info", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .label {
        text-align: center;
        margin-top: 0.5rem;
        font-size: 1.2rem;
        color: #888;
    }
    /* Center the title and subtitle */
    .centered-text {
        text-align: center;
    }
    /* Hide Streamlit sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="centered-text">Random Forest Predictor</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="centered-text">A Random Forest Algoirthm is a simple yet effective technique.'
'it blah blah blh</h3>', unsafe_allow_html=True)

