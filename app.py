import streamlit
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree, preprocessing, metrics
import math
import base64
import streamlit.components.v1 as components
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler

streamlit.set_option("deprecation.showPyplotGlobalUse", False)

# import plotly.figure_factory as ff

load_dotenv()


################ All functionalities ####################

model = XGBClassifier()
model.load_model('fraud.model') 

################# Streamlit Code Starts ####################


# Set all env variables
WCE_LOGO_PATH = os.getenv("WCE_LOGO_PATH")
WCE75 = os.getenv("WCE75")

wceLogo = Image.open(WCE_LOGO_PATH)

streamlit.set_page_config(
    page_title="Data Mining Project",
    page_icon=WCE_LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
streamlit.markdown(hide_streamlit_style, unsafe_allow_html=True)

# padding_top = 0
# padding_side = 0
# padding_bottom = 0
# streamlit.markdown(
#     f""" <style>
#     .main .block-container{{
#         padding-top: {padding_top}rem;
#         padding-right: {padding_side}rem;
#         padding-left: {padding_side}rem;
#         padding-bottom: {padding_bottom}rem;
#     }} </style> """,
#     unsafe_allow_html=True,
# )

streamlit.markdown("<br />", unsafe_allow_html=True)

cols = streamlit.columns([2, 2, 8])

with cols[1]:
    streamlit.image(wceLogo, use_column_width="auto")

with cols[2]:
    streamlit.markdown(
        """<h2 style='text-align: center; color: red'>Walchand College of Engineering, Sangli</h2>
<h6 style='text-align: center; color: black'>(An Autonomous Institute)</h6>""",
        unsafe_allow_html=True,
    )
    streamlit.markdown(
        "<h2 style='text-align: center; color: black'>Fraud Detection in Monetery Transactions</h2>",
        unsafe_allow_html=True,
    )

# with cols[3]:
#     streamlit.image(wceLogo, use_column_width='auto')
streamlit.markdown("<hr />", unsafe_allow_html=True)
# streamlit.markdown("<h3 style='text-align: center;'>Login</h3>", unsafe_allow_html=True)

styles = {
    "container": {
        "margin": "0px !important",
        "padding": "0!important",
        "align-items": "stretch",
        "background-color": "#fafafa",
    },
    "icon": {"color": "black", "font-size": "20px"},
    "nav-link": {
        "font-size": "20px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#eee",
    },
    "nav-link-selected": {
        "background-color": "lightblue",
        "font-size": "20px",
        "font-weight": "normal",
        "color": "black",
    },
}

with streamlit.sidebar:
    streamlit.markdown(
        """<h1>Guided by,</h1>
    <h3>Dr. B.F Momin<br /></h3>""",
        unsafe_allow_html=True,
    )

    streamlit.sidebar.markdown("<hr />", unsafe_allow_html=True)

    main_option = None
    dataframe = None

    main_option = option_menu(
            "",
            [
                "Data Analysis",
                "Prediction",
            ],
            icons=["clipboard-data", "eyeglasses"],
            default_index=0,
        )

    streamlit.sidebar.markdown("<hr />", unsafe_allow_html=True)

    streamlit.markdown(
        """<h2>Developed by,</h2>
    <h4>Smital Patil,<br />2019BTECS00028<hr />Suyash Chavan,</br>2019BTECS00041<hr />Aditya Sarnobat,<br/>2019BTECS00042<hr/>Krishna Mali<br/>2019BTECS00043</h4>""",
        unsafe_allow_html=True,
    )


if main_option == "Data Analysis":
    selected = option_menu(
        "",
        ["Pie Charts", "Histogram"],
        icons=["book", "eye", "search", "ui-checks"],
        orientation="horizontal",
        default_index=0,
    )

    if selected == "Pie Charts":
        streamlit.image("./analysis/pi1.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        streamlit.image("./analysis/pi2.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        streamlit.image("./analysis/pi3.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    elif selected == "Histogram":
        streamlit.image("./analysis/histo1.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        streamlit.image("./analysis/histo2.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        streamlit.image("./analysis/histo3.jpeg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    
elif main_option == "Prediction":

    cols = streamlit.columns([1,1])

    step = cols[0].number_input("Step")
    type = cols[0].selectbox("Type", ["Transfer", "Cashout"])
    amount = cols[0].number_input("Amount")
    oldbalanceOrig  = cols[0].number_input("Old Balance Original")

    newbalanceOrig = cols[1].number_input("New Balance Original")
    oldbalanceDest  = cols[1].number_input("Old Balance Destination")
    newbalanceDest = cols[1].number_input("New Balance Destination")
    isFlaggedFraud = cols[1].selectbox("Flagged Fraud", ["Yes", "No"])

    check = streamlit.button("Check")

    if check:
        if type=="Transfer":
            type = 0
        else:
            type = 1
        
        if isFlaggedFraud=="Yes":
            isFlaggedFraud = 1
        else:
            isFlaggedFraud = 0

        errorbalanceOrig  = newbalanceOrig + amount - oldbalanceOrig
        errorbalanceDest  = newbalanceDest + amount - oldbalanceDest

        x_test = [[step ,type,amount,oldbalanceOrig,newbalanceOrig,oldbalanceDest ,newbalanceDest,isFlaggedFraud,errorbalanceOrig,errorbalanceDest]]

        ans = model.predict(x_test)[0]

        if ans==0:
            streamlit.success("Not Fraud")
        else:
            streamlit.error("Fraud")







