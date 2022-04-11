import streamlit as st
#st.set_page_config(layout="wide")

# Custom imports 
from multipage import MultiPage
from pages import basic_image_processing_operations ,image_processing_basics_v3, opencv_basics

#st.set_page_config(layout="wide")

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title("Image Processing Application")

# Add all your applications (pages) here
app.add_page("Image Processing Basics V1", opencv_basics.app)
app.add_page("Image Processing Basics V2", basic_image_processing_operations.app)
app.add_page("Image Processing Basics V3", image_processing_basics_v3.app)
#app.add_page("Machine Learning", machine_learning.app)
#app.add_page("Data Analysis",data_visualize.app)
#app.add_page("Y-Parameter Optimization",redundant.app)

# The main app
app.run()