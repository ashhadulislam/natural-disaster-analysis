import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage

from pages import disasterAnalysis, detectDisaster
# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('Logo.jpg')
display = np.array(display)
st.image(display)
st.title("Satellite Image based Disaster Analysis")
st.text("Disaster Affected Or Not: To detect if an area has been hit by a disaster")
st.text("Detect Disaster Type: To detect the type of disaster affecting an area")

# col1 = st.columns(1)
# col1, col2 = st.columns(2)
# col1.image(display, width = 400)
# col2.title("Data Storyteller Application")

# Add all your application here
app.add_page("Disaster Affected Or Not", disasterAnalysis.app)
app.add_page("Detect Disaster Type", detectDisaster.app)


# The main app
app.run()
