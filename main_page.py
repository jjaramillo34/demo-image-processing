from pages import opencv_basics
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components
from utils_helpers import (
    convert_rgb_to_names,
    download_button1,
    load_image,
    increment_counter,
    load_image_PIL,
    converted,
    insert_data_mongodb,
    average_ratings_mongodb,
    get_location_data,
    tutorial,
    gitlab_code,
    scrape_duckduckgo,
    version)

# st.set_page_config(layout="wide")

# opencv_basics.app()


def welcome():
    st.markdown("<h1 style='text-align: center; color: blue;'>Welcome Page</h1>",
                unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        st.title('Image Processing Basics')
        st.image('images/image_processing.jpeg', use_column_width=True)
        st.title('Usage')
        st.markdown('A simple app that shows different image processing techniques. You can choose the options from the dropdwon menu on the left.' +
                    'Technologies use to build the app:', unsafe_allow_html=True)
        st.title('Technology Stack')
        st.markdown('''
<p align="center">
    <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
    <img src="https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white" />
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
    <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" />
    <img src="https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" />
</p>''', unsafe_allow_html=True)
    with cols[1]:
        st.title('Image Processing Techniques')
        st.markdown('''
>Loading and Displaying Images --- OpenCV Load Image (cv2.imread)
>
>Getting and Setting Pixels --- Getting and Setting Pixels
>
>Drawing with OpenCV --- Drawing with OpenCV
>
>Translation --- OpenCV Image Translation
>
>Rotation --- OpenCV Rotate Image
>
>Resizing --- OpenCV Resize Image (cv.resize)
>
>Flipping --- OpenCV Flip Image (cv.flip)
>
>Cropping --- Crop Image with OpenCV
>
>Image Arithmetic --- Image Arithmetic OpenCV
>
>Bitwise Operations --- OpenCV Bitwise AND, OR, XOR, and NOT''', unsafe_allow_html=True)
        st.title('Dedication')
        st.markdown('''> To my Mother (Elsa), Paula, Cris, Maty and Sofia, To whom made this possible.
>
> Special thanks to Adrian from pyimagesearch.com for great tutorials of image processing, deep learning, augmented realty, etc. ''')

        st.markdown('''> Long Live Rock N Roll.
>
> - "Well if I have to, I will die seven deaths just to lie In the arms of my eversleeping aim"''')
        st.title('Contact')
        st.markdown('''<p align="center">
    <a href="mailto:jjaramillo34@gmail.com" rel="nofollow">
        <img alt="Gmail" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
    </a>
    <a href="https://github.com/jjaramillo34/" rel="nofollow">
        <img alt="Github" src="https://img.shields.io/badge/GitHub-%2312100E.svg?&style=for-the-badge&logo=Github&logoColor=white"/>
    </a>
    <a href="https://twitter.com/jejaramilloc" rel="nofollow">
        <img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white"/>
    </a>
    <a href="https://www.linkedin.com/in/javierjaramillo1/" rel="nofollow">
        <img alt="Linkedin" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
    </a>
    </p>''', unsafe_allow_html=True)

    location_dict = get_location_data()

    print(location_dict)

    date_r = datetime.now()
    city = location_dict['city']
    ip = location_dict['ip']
    region = location_dict['region']
    country = location_dict['country']
    loc = location_dict['loc']

    # set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    with st.sidebar.form(key='columns_in_form', clear_on_submit=True):
        rating = st.slider("Please rate the app", min_value=1, max_value=5, value=3,
                           help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
        feedback = st.text_input(label='Please leave your feedback here')
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your feedback!')
            st.markdown('Your Rating:')
            st.markdown(rating)
            st.markdown('Your Feedback:')
            st.markdown(feedback)
            insert_data_mongodb(rating=rating, feedback=feedback, date_r=date_r,
                                city=city, ip=ip, region=region, country=country, loc=loc)

    score_average = average_ratings_mongodb()
    if score_average == 5.0:
        st.sidebar.title('App Ratings')
        st.sidebar.markdown(
            f'⭐⭐⭐⭐⭐ <p style="font-weight:bold;color:green;font-size:20px;border-radius:2%;">{round(score_average, 1)}</p>', unsafe_allow_html=True)
    elif score_average >= 4.0 and score_average < 5.0:
        st.sidebar.title('App Ratings')
        st.sidebar.markdown(
            f'⭐⭐⭐⭐ <p style="font-weight:bold;color:green;font-size:20px;border-radius:2%;">{round(score_average, 1)}</p>', unsafe_allow_html=True)
    elif score_average >= 3.0 and score_average < 4.0:
        st.sidebar.title('App Ratings')
        st.sidebar.markdown(
            f'⭐⭐⭐ <p style="font-weight:bold;color:green;font-size:20px;border-radius:2%;">{round(score_average, 1)}</p>', unsafe_allow_html=True)
    elif score_average >= 2.0 and score_average < 3.0:
        st.sidebar.title('App Ratings')
        st.sidebar.markdown(
            f'⭐⭐ <p style="font-weight:bold;color:green;font-size:20px;border-radius:2%;">{round(score_average, 1)}</p>', unsafe_allow_html=True)
    elif score_average < 2.0:
        st.sidebar.title('App Ratings')
        st.sidebar.markdown(
            f'⭐ <p style="font-weight:bold;color:green;font-size:20px;border-radius:2%;">{round(score_average, 1)}</p>', unsafe_allow_html=True)

    st.sidebar.markdown(
        f'<p style="font-weight:bold;color:black;font-size:12px;border-radius:2%;">Ratings live atlas mongodb database feed</p>', unsafe_allow_html=True)

    with st.expander('Show MongoDB Dashboard', expanded=True):
        components.iframe(
            'https://charts.mongodb.com/charts-project-0-koqvp/public/dashboards/62523657-6131-48ab-8c6c-3893cfb849fa', height=900)

    version()

# Custom imports
#from multipage import MultiPage
#from pages import basic_image_processing_operations ,image_processing_basics_v3, opencv_basics

# st.set_page_config(layout="wide")

# Create an instance of the app
#app = MultiPage()

# Title of the main page
#st.title("Image Processing Application")

# Add all your applications (pages) here
#app.add_page("Image Processing Basics V1", opencv_basics.app)
#app.add_page("Image Processing Basics V2", basic_image_processing_operations.app)
#app.add_page("Image Processing Basics V3", image_processing_basics_v3.app)
#app.add_page("Machine Learning", machine_learning.app)
#app.add_page("Data Analysis",data_visualize.app)
#app.add_page("Y-Parameter Optimization",redundant.app)

# The main app
# app.run()


welcome()
