from ast import Not
from lib2to3.pytree import convert
from trace import CoverageResults
from pandas import options
import streamlit as st
import cv2 as cv
import numpy as np
import string
import random
from io import BytesIO
import requests
import shutil
import imutils
import streamlit.components.v1 as components
from datetime import datetime
from streamlit_cropper import st_cropper
from webcolors import hex_to_name
from PIL import Image, ImageColor
from matplotlib import pyplot as plt
from utils_helpers import (
    auto_canny_thresh,
    source_code,
    version,
    load_image,
    load_image_PIL,
    converted,
    # download_button,
    get_location_data,
    download_button1,
    convolve,
    insert_data_mongodb,
    average_ratings_mongodb,
    source_code,
    scrape_duckduckgo)

selected_boxes = (
    "Welcome",
    "Demo Adaptive Thresholding",
    "Demo Auto Canny",
    "Demo Canny Edge Detector",
    "Demo Convolutions",
    "Demo Image Gradients",
    "Demo Morphological Operations",
    "Demo Color Spaces",
    "Demo Color Thresholding",
    "Demo Smoothing and Blurring",
)

rand = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
download = f'{rand}.jpeg'

language = 'python'
default_image = 'images/nice.jpeg'
button = 'Download Result Image'
original = 'Original Image'
code = 'Source Code'
mime_type = 'image/jpeg'
font = cv.FONT_HERSHEY_SIMPLEX


def app():

    selected_box = st.sidebar.selectbox(
        "Choosse on of the following", selected_boxes)

    if selected_box == "Welcome":
        welcome()
    if selected_box == "Demo Adaptive Thresholding":
        adaptive_thresholding()
    if selected_box == "Demo Auto Canny":
        auto_canny()
    if selected_box == "Demo Canny Edge Detector":
        canny_edge_detector()
    if selected_box == "Demo Convolutions":
        convolutions()
    if selected_box == "Demo Image Gradients":
        image_gradients()
    if selected_box == "Demo Morphological Operations":
        morphological_operations()
    if selected_box == "Demo Color Thresholding":
        thresholding()
    if selected_box == "Demo Color Spaces":
        color_spaces()
    if selected_box == "Demo Smoothing and Blurring":
        smoothing_blurring()


def welcome():
    cols = st.columns(2)
    with cols[0]:
        st.title('Basic Image Processing Operations')
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
>Morphological Operations --- OpenCV Morphological Operations
>
>Smoothing and Blurring ---  OpenCV Smoothing and Blurring
>
>Color Spaces -- OpenCV Color Spaces (cv2.cvtColor)
>
>Basic Thresholding --- OpenCV Thresholding (cv2.threshold) 
>
>Adaptive Thresholding --- Adaptive Thresholding with OpenCV (cv2.adaptiveThreshold)
>
>Kernels --- Convolutions with OpenCV and Python
>
>Image Gradients --- Image Gradients with OpenCV (Sobel and Scharr)
>
>Edge Detection --- OpenCV Edge Detection (cv2.Canny)
>
>Automatic Edge Detection --- Zero-parameter, automatic Canny edge detection with Python and OpenCV''', unsafe_allow_html=True)
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

    with st.expander('Show MongoDB Dashboard'):
        components.iframe(
            'https://charts.mongodb.com/charts-project-0-koqvp/public/dashboards/62523657-6131-48ab-8c6c-3893cfb849fa', height=800)

    version()


def adaptive_thresholding():

    st.header("Demo Adaptive Thresholding")
    options = st.sidebar.radio('Adaptive Thresholding Options',
                               ('Adaptive Thresholding', 'Adaptive Thesholding Interactive'))
    if options == 'Adaptive Thresholding':
        img_file = st.file_uploader(label='Upload a file', type=[
                                    'png', 'jpg', 'jpge'], key='1')
        if img_file is not None:
            # load the image and display it
            with st.expander('Show Original Image'):
                image = load_image_PIL(img_file)
                image = converted(image)
                # convert the image to grayscale and blur it slightly
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                blurred = cv.GaussianBlur(gray, (7, 7), 0)
                st.image(image)

            with st.expander('Show Adaptive Thresholding', expanded=True):
                cols = st.columns(4)
                (T, threshInv) = cv.threshold(blurred, 51, 255,
                                              cv.THRESH_BINARY_INV)
                cols[0].markdown("Simple Thresholding")
                cols[0].image(threshInv)
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.1')

                # apply Otsu's automatic thresholding
                (T, threshInv) = cv.threshold(blurred, 0, 255,
                                              cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
                cols[1].markdown("Otsu Thresholding")
                cols[1].image(threshInv)
                with cols[1]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.2')

                # instead of manually specifying the threshold value, we can use adaptive thresholding to examine neighborhoods
                # of pixels and adaptively threshold each neighborhood
                thresh = cv.adaptiveThreshold(blurred, 255,
                                              cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)
                cols[2].markdown("Mean Adaptive Thresholding")
                cols[2].image(threshInv)
                with cols[2]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.3')

                # perform adaptive thresholding again, this time using a Gaussian weighting versus a simple mean to compute our
                # local threshold value
                thresh = cv.adaptiveThreshold(blurred, 255,
                                              cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)
                cols[3].markdown("Gaussian Adaptive Thresholding")
                cols[3].image(thresh)
                with cols[3]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.4')

            with st.expander("Show Adaptive Thresholding Types Interactive"):
                x = st.slider('Change Threshold value',
                              min_value=50, max_value=255, key='1')

                ret, thresh1 = cv.threshold(blurred, x, 255, cv.THRESH_BINARY)
                ret, thresh2 = cv.threshold(
                    blurred, x, 255, cv.THRESH_BINARY_INV)
                ret, thresh3 = cv.threshold(blurred, x, 255, cv.THRESH_TRUNC)
                ret, thresh4 = cv.threshold(blurred, x, 255, cv.THRESH_TOZERO)
                ret, thresh5 = cv.threshold(
                    blurred, x, 255, cv.THRESH_TOZERO_INV)
                titles = ['Original Image', 'BINARY',
                          'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
                images = [blurred, thresh1, thresh2, thresh3, thresh4, thresh5]

                cols = st.columns(3)
                for i in range(0, 3):
                    cols[i].markdown(i)
                    cols[i].markdown(titles[i])
                    cols[i].image(images[i])
                    with cols[i]:
                        download_button1(
                            images[i], button, download, mime_type, key='{i}.1.1')

                cols = st.columns(3)
                for i in range(3, 6):
                    cols[i-3].markdown(i)
                    cols[i-3].markdown(titles[i])
                    cols[i-3].image(images[i])
                    with cols[i - 3]:
                        download_button1(
                            images[i], button, download, mime_type, key='{i}.2.2')

        else:
            # load the image and display it
            with st.expander('Show Original Image'):
                image = load_image('images/steve-jobs.jpg')
                # convert the image to grayscale and blur it slightly
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                blurred = cv.GaussianBlur(gray, (7, 7), 0)
                st.image(image)

            with st.expander('Show Adaptive Thresholding', expanded=True):
                cols = st.columns(4)
                (T, threshInv) = cv.threshold(blurred, 51, 255,
                                              cv.THRESH_BINARY_INV)
                cols[0].markdown("Simple Thresholding")
                cols[0].image(threshInv)
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.1')

                # apply Otsu's automatic thresholding
                (T, threshInv) = cv.threshold(blurred, 0, 255,
                                              cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
                cols[1].markdown("Otsu Thresholding")
                cols[1].image(threshInv)
                with cols[1]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.2')

                # instead of manually specifying the threshold value, we can use adaptive thresholding to examine neighborhoods
                # of pixels and adaptively threshold each neighborhood
                thresh = cv.adaptiveThreshold(blurred, 255,
                                              cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)
                cols[2].markdown("Mean Adaptive Thresholding")
                cols[2].image(threshInv)
                with cols[2]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.3')

                # perform adaptive thresholding again, this time using a Gaussian weighting versus a simple mean to compute our
                # local threshold value
                thresh = cv.adaptiveThreshold(blurred, 255,
                                              cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 4)
                cols[3].markdown("Gaussian Adaptive Thresholding")
                cols[3].image(thresh)
                with cols[3]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.4')

            with st.expander("Show Adaptive Thresholding Types Interactive"):
                x = st.slider('Change Threshold value',
                              min_value=50, max_value=255, key='adaptive_1.1')

                ret, thresh1 = cv.threshold(blurred, x, 255, cv.THRESH_BINARY)
                ret, thresh2 = cv.threshold(
                    blurred, x, 255, cv.THRESH_BINARY_INV)
                ret, thresh3 = cv.threshold(blurred, x, 255, cv.THRESH_TRUNC)
                ret, thresh4 = cv.threshold(blurred, x, 255, cv.THRESH_TOZERO)
                ret, thresh5 = cv.threshold(
                    blurred, x, 255, cv.THRESH_TOZERO_INV)
                titles = ['Original Image', 'BINARY',
                          'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
                images = [blurred, thresh1, thresh2, thresh3, thresh4, thresh5]

                cols = st.columns(3)
                for i in range(0, 3):
                    cols[i].markdown(i)
                    cols[i].markdown(titles[i])
                    cols[i].image(images[i])
                    with cols[i]:
                        download_button1(
                            images[i], button, download, mime_type, key='{i}.1.1')

                cols = st.columns(3)
                for i in range(3, 6):
                    cols[i-3].markdown(i)
                    cols[i-3].markdown(titles[i])
                    cols[i-3].image(images[i])
                    with cols[i - 3]:
                        download_button1(
                            images[i], button, download, mime_type, key='{i}.2.2')

    else:
        img_file = st.file_uploader(label='Upload a file', type=[
                                    'png', 'jpg', 'jpge'], key='1')
        if img_file is not None:
            with st.expander('Show Original Image'):
                image = load_image_PIL(img_file)
                image = converted(image)
                # convert the image to grayscale and blur it slightly
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                blurred = cv.GaussianBlur(gray, (7, 7), 0)
                st.image(image)

            with st.expander('Show Adaptive Thresholding Interactive', expanded=True):
                cols = st.columns(4)
                x = cols[0].slider('Change Threshold value',
                                   min_value=50, max_value=255, key='1')
                (T, threshInv) = cv.threshold(blurred, x, 255,
                                              cv.THRESH_BINARY_INV)
                cols[0].markdown('Simple Thresholding')
                cols[0].image(threshInv, use_column_width=True, clamp=True)
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.1')

                x = cols[1].slider('Change Threshold value', min_value=50,
                                   max_value=255, key='2', help='Auto threshold value selected')
                # apply Otsu's automatic thresholding
                (T, threshInv) = cv.threshold(blurred, 0, 255,
                                              cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
                cols[1].markdown("Otsu's Automatic Thresholding")
                cols[1].image(threshInv, use_column_width=True, clamp=True)
                with cols[1]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.2')

                x = cols[2].slider('Change Threshold value',
                                   min_value=21, max_value=255, step=2, key='3')
                # instead of manually specifying the threshold value, we can use adaptive thresholding to examine neighborhoods
                # of pixels and adaptively threshold each neighborhood
                thresh = cv.adaptiveThreshold(blurred, 255,
                                              cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, x, 10)
                cols[2].markdown('Mean Adaptive Thresholding')
                cols[2].image(thresh, use_column_width=True, clamp=True)
                with cols[2]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.3')

                x = cols[3].slider('Change Threshold value',
                                   min_value=21, max_value=255, step=2, key='4')
                # perform adaptive thresholding again, this time using a Gaussian weighting versus a simple mean to compute our
                # local threshold value
                thresh = cv.adaptiveThreshold(blurred, 255,
                                              cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, x, 4)
                cols[3].markdown('Gaussian Adaptive Thresholding')
                cols[3].image(thresh, use_column_width=True, clamp=True)
                cols[3].text("Bar Chart of the image")
                with cols[3]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.4')
        else:
            with st.expander('Show Original Image'):
                image = load_image('images/steve-jobs.jpg')
                # convert the image to grayscale and blur it slightly
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                blurred = cv.GaussianBlur(gray, (7, 7), 0)
                st.image(image)

            with st.expander('Show Adaptive Thresholding Interactive', expanded=True):
                cols = st.columns(4)
                x = cols[0].slider('Change Threshold value',
                                   min_value=50, max_value=255, key='1')
                (T, threshInv) = cv.threshold(blurred, x, 255,
                                              cv.THRESH_BINARY_INV)
                cols[0].markdown('Simple Thresholding')
                cols[0].image(threshInv, use_column_width=True, clamp=True)
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.1')

                x = cols[1].slider('Change Threshold value', min_value=50, max_value=255,
                                   key='2', disabled=True, help='Auto threshold value selected')
                # apply Otsu's automatic thresholding
                (T, threshInv) = cv.threshold(blurred, 0, 255,
                                              cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
                cols[1].markdown("Otsu's Automatic Thresholding")
                cols[1].image(threshInv, use_column_width=True, clamp=True)
                with cols[1]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.2')

                x = cols[2].slider('Change Threshold value',
                                   min_value=21, max_value=255, step=2, key='3')
                # instead of manually specifying the threshold value, we can use adaptive thresholding to examine neighborhoods
                # of pixels and adaptively threshold each neighborhood
                thresh = cv.adaptiveThreshold(blurred, 255,
                                              cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, x, 10)
                cols[2].markdown('Mean Adaptive Thresholding')
                cols[2].image(thresh, use_column_width=True, clamp=True)
                with cols[2]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.3')

                x = cols[3].slider('Change Threshold value',
                                   min_value=21, max_value=255, step=2, key='4')
                # perform adaptive thresholding again, this time using a Gaussian weighting versus a simple mean to compute our
                # local threshold value
                thresh = cv.adaptiveThreshold(blurred, 255,
                                              cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, x, 4)
                cols[3].markdown('Gaussian Adaptive Thresholding')
                cols[3].image(thresh, use_column_width=True, clamp=True)
                cols[3].text("Bar Chart of the image")
                with cols[3]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.4')

    source_code(
        'Source Code + Adaptive Thresholding pyimagesearch.com',
        'https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/',
        'https://gist.github.com/jjaramillo34/331a1aaeebeb4ff47d9b80a658643b60')

    with st.expander('DuckDuckGo Search Results'):
        st.subheader('More About Adaptive Thresholding')
        #scrape_duckduckgo('adaptive thresholding opencv')
        scrape_duckduckgo('adaptive thresholding opencv')


def auto_canny():
    st.header("Auto Canny Demo")
    realtime_update = st.sidebar.checkbox(
        label="Update in Real Time", value=True)
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpeg'])
    if img_file is not None:
        # load the image, convert it to grayscale, and blur it slightly
        image = load_image_PIL(img_file)
        image = converted(image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (3, 3), 0)

        # apply Canny edge detection using a wide threshold, tight threshold, and automatically determined threshold
        wide = cv.Canny(blurred, 10, 200)
        tight = cv.Canny(blurred, 225, 250)
        auto = auto_canny_thresh(blurred)

        images = [wide, tight, auto]
        labels = ['Wide Edges', 'Tight Edges', 'Auto Canny']
        # show the images
        with st.expander('Show Original Image'):
            st.markdown("Original")
            st.image(image)

        with st.expander('Show Auto Canny', expanded=True):
            cols = st.columns(3)
            for i, image in enumerate(images):
                cols[i].markdown(labels[i])
                cols[i].image(image)
                with cols[i]:
                    download_button1(image, button, download,
                                     mime_type, key=f'{i}.1')
    else:
        # load the image, convert it to grayscale, and blur it slightly
        image = load_image(default_image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (3, 3), 0)

        # apply Canny edge detection using a wide threshold, tight
        # threshold, and automatically determined threshold
        wide = cv.Canny(blurred, 10, 200)
        tight = cv.Canny(blurred, 225, 250)
        auto = auto_canny_thresh(blurred)

        images = [wide, tight, auto]
        labels = ['Wide Edges', 'Tight Edges', 'Auto Canny']
        # show the images
        with st.expander('Show Original Image'):
            st.markdown("Original")
            st.image(image)

        with st.expander('Show Auto Canny', expanded=True):
            cols = st.columns(3)
            for i, image in enumerate(images):
                cols[i].markdown(labels[i])
                cols[i].image(image)
                with cols[i]:
                    download_button1(image, button, download,
                                     mime_type, key=f'{i}.1')

    source_code(
        'Source Code + Auto Canny pyimagesearch.com',
        'https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/',
        'https://gist.github.com/jjaramillo34/fb83acff62ce6502c398ba7133ab066c')

    with st.expander('DuckDuckGo Search Results'):
        st.subheader('More About Auto Canny')
        scrape_duckduckgo('auto canny opencv')


def convolutions():
    st.header("Resizing Demo")
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpge'])
    realtime_update = st.sidebar.checkbox(
        label="Update in Real Time", value=True)

    if img_file is not None:
        image = load_image_PIL(img_file)
        image = converted(image)

    else:
        # construct average blurring kernels used to smooth an image
        smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
        largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

        # construct a sharpening filter
        sharpen = np.array((
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]), dtype="int")

        # construct the Laplacian kernel used to detect edge-like regions of an image
        laplacian = np.array((
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]), dtype="int")

        # construct the Sobel x-axis kernel
        sobelX = np.array((
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]), dtype="int")

        # construct the Sobel y-axis kernel
        sobelY = np.array((
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]), dtype="int")

        # construct the kernel bank, a list of kernels we're going to apply using both our
        # custom `convole` function and OpenCV's `filter2D` function
        kernelBank = (
            ("small_blur", smallBlur),
            ("large_blur", largeBlur),
            ("sharpen", sharpen),
            ("laplacian", laplacian),
            ("sobel_x", sobelX),
            ("sobel_y", sobelY)
        )

        # load the input image and convert it to grayscale
        image = load_image('images/supermario.jpg')
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # loop over the kernels
        with st.spinner('Creating Convolutions please wait for it...'):
            for (kernelName, kernel) in kernelBank:
                # apply the kernel to the grayscale image using both our custom 'convole'
                # function and OpenCV's 'filter2D' function
                st.write("[INFO] applying {} kernel".format(kernelName))
                convoleOutput = convolve(gray, kernel)
                opencvOutput = cv.filter2D(gray, -1, kernel)

                # show the output images
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("original")
                    st.image(gray)
                with col2:
                    st.write("{} - convole".format(kernelName))
                    st.image(convoleOutput)
                with col3:
                    st.write("{} - opencv".format(kernelName))
                    st.image(opencvOutput)
            st.success('Convolutions were created succesfully!')
        col1, col2 = st.columns(2)
        with st.expander('Source Code'):
            with col1:
                st.markdown(code)
                st.code('''
# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")

# construct the kernel bank, a list of kernels we're going to apply using both our 
# custom `convole` function and OpenCV's `filter2D` function
kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY)
)

# load the input image and convert it to grayscale
image = load_image('images/supermario.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# loop over the kernels
with st.spinner('Creating Convolutions please wait for it...'):
    for (kernelName, kernel) in kernelBank:
        # apply the kernel to the grayscale image using both our custom 'convole' 
        # function and OpenCV's 'filter2D' function
        st.write("[INFO] applying {} kernel".format(kernelName))
        convoleOutput = convolve(gray, kernel)
        opencvOutput = cv.filter2D(gray, -1, kernel)

        # show the output images
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("original")
            st.image(gray)
        with col2:
            st.write("{} - convole".format(kernelName)) 
            st.image(convoleOutput)
        with col3:
            st.write("{} - opencv".format(kernelName))
            st.image(opencvOutput)''', language=language)

            with col2:
                st.markdown('Source Code Convole Function')
                st.code('''
def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return the output image
    return output''', language=language)

        with st.expander('Convolutions with OpenCV and Python'):
            # embed streamlit docs in a streamlit app
            components.iframe(
                "https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/", height=800)

        version()


def canny_edge_detector():
    st.header("Canny Edge Detector Demo")
    img_file = st.file_uploader(label='Upload a file', type=[
                                'png', 'jpg', 'jpge'], key='1')
    realtime_update = st.sidebar.checkbox(
        label="Update in Real Time", value=True)
    if img_file is not None:
        image = load_image_PIL(img_file)
        image = converted(image)
        options = st.sidebar.radio('Canny Edge Detector Options',
                                   ('Canny Edge Detector', 'Canny Edge Detector Interactive'))
        if options == 'Canny Edge Detector':
            # load the image, convert it to grayscale, and blur it slightly
            #image = load_image(default_image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (5, 5), 0)

            col1, col2 = st.columns(2)
            with col1:
                with st.expander('Show Original Image'):
                    # show the original and blurred images
                    st.markdown("Original")
                    st.image(image)
            with col2:
                with st.expander('Show Blurred Image'):
                    st.markdown("Blurred")
                    st.image(blurred)

            # compute a "wide", "mid-range", and "tight" threshold for the edges
            # using the Canny edge detector
            wide = cv.Canny(blurred, 10, 200)
            mid = cv.Canny(blurred, 30, 150)
            tight = cv.Canny(blurred, 240, 250)

            col1, col2, col3 = st.columns(3)
            # show the output Canny edge maps
            with col1:
                st.markdown("Wide Edge Map")
                st.image(wide)
            with col2:
                st.markdown("Mid Edge Map")
                st.image(mid)
            with col3:
                st.markdown("Tight Edge Map")
                st.image(tight)

        else:
            image = load_image_PIL(img_file)
            image = converted(image)
            # load the image, convert it to grayscale, and blur it slightly
            image = load_image(default_image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (5, 5), 0)

            col1, col2 = st.columns(2)
            with col1:
                with st.expander('Show Original Image'):
                    # show the original and blurred images
                    st.markdown("Original")
                    st.image(image)
            with col2:
                with st.expander('Show Blurred Image'):
                    st.markdown("Blurred")
                    st.image(blurred)

            # compute a "wide", "mid-range", and "tight" threshold for the edges
            # using the Canny edge detector
            col1, col2, col3 = st.columns(3)
            # show the output Canny edge maps
            with col1:
                values = st.slider(
                    'Select a range of values',
                    10, 200, (10, 200), step=10)
                wide = cv.Canny(blurred, values[0], values[1])
                st.markdown("Wide Edge Map")
                st.image(wide)
            with col2:
                values = st.slider(
                    'Select a range of values',
                    30, 150, (30, 150), step=5)
                mid = cv.Canny(blurred, values[0], values[1])
                st.markdown("Mid Edge Map")
                st.image(mid)
            with col3:
                values = st.slider(
                    'Select a range of values',
                    200, 250, (200, 250))
                tight = cv.Canny(blurred, values[0], values[1])
                st.markdown("Tight Edge Map")
                st.image(tight)
    else:
        options = st.sidebar.radio('Canny Edge Detector Options',
                                   ('Canny Edge Detector', 'Canny Edge Detector Interactive'))
        if options == 'Canny Edge Detector':
            # load the image, convert it to grayscale, and blur it slightly
            image = load_image(default_image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (5, 5), 0)

            col1, col2 = st.columns(2)
            with col1:
                with st.expander('Show Original Image'):
                    # show the original and blurred images
                    st.markdown("Original")
                    st.image(image)
            with col2:
                with st.expander('Show Blurred Image'):
                    st.markdown("Blurred")
                    st.image(blurred)

            # compute a "wide", "mid-range", and "tight" threshold for the edges
            # using the Canny edge detector
            wide = cv.Canny(blurred, 10, 200)
            mid = cv.Canny(blurred, 30, 150)
            tight = cv.Canny(blurred, 240, 250)

            col1, col2, col3 = st.columns(3)
            # show the output Canny edge maps
            with col1:
                st.markdown("Wide Edge Map")
                st.image(wide)
            with col2:
                st.markdown("Mid Edge Map")
                st.image(mid)
            with col3:
                st.markdown("Tight Edge Map")
                st.image(tight)
        else:
            # load the image, convert it to grayscale, and blur it slightly
            image = load_image(default_image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (5, 5), 0)

            col1, col2 = st.columns(2)
            with col1:
                with st.expander('Show Original Image'):
                    # show the original and blurred images
                    st.markdown("Original")
                    st.image(image)
            with col2:
                with st.expander('Show Blurred Image'):
                    st.markdown("Blurred")
                    st.image(blurred)

            # compute a "wide", "mid-range", and "tight" threshold for the edges
            # using the Canny edge detector
            col1, col2, col3 = st.columns(3)
            # show the output Canny edge maps
            with col1:
                values = st.slider(
                    'Select a range of values',
                    10, 200, (10, 200), step=10)
                wide = cv.Canny(blurred, values[0], values[1])
                st.markdown("Wide Edge Map")
                st.image(wide)
            with col2:
                values = st.slider(
                    'Select a range of values',
                    30, 150, (30, 150), step=5)
                mid = cv.Canny(blurred, values[0], values[1])
                st.markdown("Mid Edge Map")
                st.image(mid)
            with col3:
                values = st.slider(
                    'Select a range of values',
                    200, 250, (200, 250))
                tight = cv.Canny(blurred, values[0], values[1])
                st.markdown("Tight Edge Map")
                st.image(tight)


def image_gradients():
    st.header("Image Gradient Demo")

    options = st.sidebar.radio(
        'Image Gradient Options', ('Sobel/Scharr', 'Magnitude Orientation'))

    if options == 'Sobel/Scharr':
        img_file = st.file_uploader(
            label='Upload a file', type=['png', 'jpg', 'jpge'])
        realtime_update = st.sidebar.checkbox(
            label="Update in Real Time", value=True)

        if img_file is not None:
            # load the image, convert it to grayscale, and display the original
            # grayscale image
            with st.expander('Show Sobel/Scharr Image Gradient', expanded=True):
                image = load_image_PIL(img_file)
                image = converted(image)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                # set the kernel size, depending on whether we are using the Sobel operator of the Scharr operator,
                # then compute the gradients along the x and y axis, respectively
                op = st.selectbox('Operators', ('sobel', 'scharr'))
                if op == 'scharr':
                    s = 1
                else:
                    s = 0

                st.success(f'Operator Selected: {op}')

                ksize = -1 if s > 0 else 3
                gX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=ksize)
                gY = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=ksize)

                # the gradient magnitude images are now of the floating point data type, so we need to take care
                # to convert them back a to unsigned 8-bit integer representation so other OpenCV functions can
                # operate on them and visualize them
                gX = cv.convertScaleAbs(gX)
                gY = cv.convertScaleAbs(gY)

                # combine the gradient representations into a single image
                combined = cv.addWeighted(gX, 0.5, gY, 0.5, 0)

                # show our output images
                cols = st.columns(4)
                cols[0].markdown("Gray")
                cols[0].image(gray)
                cols[1].markdown("Sobel/Scharr X")
                cols[1].image(gX)
                with cols[1]:
                    download_button1(gX, button, download,
                                     mime_type, key='1.1')
                cols[2].markdown("Sobel/Scharr Y")
                cols[2].image(gY)
                with cols[2]:
                    download_button1(gY, button, download,
                                     mime_type, key='1.2')
                cols[3].markdown("Sobel/Scharr Combined")
                cols[3].image(combined)
                with cols[3]:
                    download_button1(combined, button,
                                     download, mime_type, key='1.3')

        else:
            # load the image, convert it to grayscale, and display the original
            # grayscale image
            with st.expander('Show Sobel/Scharr Image Gradient', expanded=True):
                image = load_image(default_image)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                # set the kernel size, depending on whether we are using the Sobel operator of the Scharr operator,
                # then compute the gradients along the x and y axis, respectively
                op = st.selectbox('Operators', ('sobel', 'scharr'))
                if op == 'scharr':
                    s = 1
                else:
                    s = 0

                st.success(f'Operator Selected: {op}')

                ksize = -1 if s > 0 else 3
                gX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=ksize)
                gY = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=ksize)

                # the gradient magnitude images are now of the floating point data type, so we need to take care
                # to convert them back a to unsigned 8-bit integer representation so other OpenCV functions can
                # operate on them and visualize them
                gX = cv.convertScaleAbs(gX)
                gY = cv.convertScaleAbs(gY)

                # combine the gradient representations into a single image
                combined = cv.addWeighted(gX, 0.5, gY, 0.5, 0)

                # show our output images
                cols = st.columns(4)
                cols[0].markdown("Gray")
                cols[0].image(gray)
                cols[1].markdown("Sobel/Scharr X")
                cols[1].image(gX)
                with cols[1]:
                    download_button1(gX, button, download,
                                     mime_type, key='2.1')
                cols[2].markdown("Sobel/Scharr Y")
                cols[2].image(gY)
                with cols[2]:
                    download_button1(gY, button, download,
                                     mime_type, key='2.2')
                cols[3].markdown("Sobel/Scharr Combined")
                cols[3].image(combined)
                with cols[3]:
                    download_button1(combined, button,
                                     download, mime_type, key='2.3')

    else:
        img_file = st.file_uploader(
            label='Upload a file', type=['png', 'jpg', 'jpge'])
        realtime_update = st.sidebar.checkbox(
            label="Update in Real Time", value=True)
        if img_file is not None:
            # load the input image and convert it to grayscale
            image = load_image_PIL(img_file)
            image = converted(image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # compute gradients along the x and y axis, respectively
            gX = cv.Sobel(gray, cv.CV_64F, 1, 0)
            gY = cv.Sobel(gray, cv.CV_64F, 0, 1)

            # compute the gradient magnitude and orientation
            magnitude = np.sqrt((gX ** 2) + (gY ** 2))
            orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

            #arr = np.uint8(magnitude)
            dist1 = cv.convertScaleAbs(magnitude)
            imC1 = cv.applyColorMap(dist1, cv.COLORMAP_JET)

            dist2 = cv.convertScaleAbs(orientation)
            imC2 = cv.applyColorMap(dist2, cv.COLORMAP_JET)

            # display all images
            with st.expander('Show Magnitude - Orientation Image Gradients - Streamlit', expanded=True):
                cols = st.columns(3)
                cols[0].markdown("Grayscale")
                cols[0].image(gray)
                cols[1].markdown("Gradient Magnitude")
                cols[1].image(imC1, channels='BGR')
                with cols[1]:
                    download_button1(imC1, button, download,
                                     mime_type, key='3.1')
                cols[2].markdown("Gradient Orientation [0, 180]")
                cols[2].image(imC2, channels='BGR')
                with cols[2]:
                    download_button1(imC2, button, download,
                                     mime_type, key='3.2')

            # initialize a figure to display the input grayscle image along with
            # the gradient magnitude and orientation representations, respectively
            (fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))

            # plot each of the images
            axs[0].imshow(gray, cmap="gray")
            axs[1].imshow(magnitude, cmap="jet")
            axs[2].imshow(orientation, cmap="jet")

            # set the titles of each axes
            axs[0].set_title("Grayscale")
            axs[1].set_title("Gradient Magnitude")
            axs[2].set_title("Gradient Orientation [0, 180]")

            # loop over each of the axes and turn off the x and y ticks
            for i in range(0, 3):
                axs[i].get_xaxis().set_ticks([])
                axs[i].get_yaxis().set_ticks([])

            with st.expander('Show Magnitude - Orientation Image Gradients - Mapplotlib'):
                # show the plots
                plt.tight_layout()
                st.pyplot(fig)

        else:
            # load the input image and convert it to grayscale
            image = load_image(default_image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # compute gradients along the x and y axis, respectively
            gX = cv.Sobel(gray, cv.CV_64F, 1, 0)
            gY = cv.Sobel(gray, cv.CV_64F, 0, 1)

            # compute the gradient magnitude and orientation
            magnitude = np.sqrt((gX ** 2) + (gY ** 2))
            orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

            #arr = np.uint8(magnitude)
            dist1 = cv.convertScaleAbs(magnitude)
            imC1 = cv.applyColorMap(dist1, cv.COLORMAP_OCEAN)

            dist2 = cv.convertScaleAbs(orientation)
            imC2 = cv.applyColorMap(dist2, cv.COLORMAP_JET)

            # display all images
            with st.expander('Show Magnitude - Orientation Image Gradients - Streamlit', expanded=True):
                cols = st.columns(3)
                cols[0].markdown("Grayscale")
                cols[0].image(gray)
                cols[1].markdown("Gradient Magnitude")
                cols[1].image(imC1, clamp=False)
                with cols[1]:
                    download_button1(imC1, button, download,
                                     mime_type, key='5.1')
                cols[2].markdown("Gradient Orientation [0, 180]")
                cols[2].image(imC2, clamp=True)
                with cols[2]:
                    download_button1(imC2, button, download,
                                     mime_type, key='5.2')

            # initialize a figure to display the input grayscle image along with
            # the gradient magnitude and orientation representations, respectively
            (fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))

            # plot each of the images
            axs[0].imshow(gray, cmap="gray")
            axs[1].imshow(magnitude, cmap="jet")
            axs[2].imshow(orientation, cmap="jet")

            # set the titles of each axes
            axs[0].set_title("Grayscale")
            axs[1].set_title("Gradient Magnitude")
            axs[2].set_title("Gradient Orientation [0, 180]")

            # loop over each of the axes and turn off the x and y ticks
            for i in range(0, 3):
                axs[i].get_xaxis().set_ticks([])
                axs[i].get_yaxis().set_ticks([])

            with st.expander('Show Magnitude - Orientation Image Gradients - Mapplotlib'):
                # show the plots
                plt.tight_layout()
                st.pyplot(fig)

    source_code(
        'Source Code + Image Gradients Tutorial pyimagesearch.com',
        'https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/',
        'https://gist.github.com/jjaramillo34/4a40d2faeddda4c1275b2c40c86260a4')

    with st.expander('DuckDuckGo Search Results'):
        st.subheader('More About Morphological Operations')
        scrape_duckduckgo('morphological operations opencv')


def morphological_operations():
    st.header("Morphological Operations Demo")

    options = st.sidebar.radio('Morphological Operations Options',
                               ('Morphological Hats', 'Morphological Operations'))

    if options == 'Morphological Hats':

        img_file = st.file_uploader(
            label='Upload a file', type=['png', 'jpg', 'jpge'])
        realtime_update = st.sidebar.checkbox(
            label="Update in Real Time", value=True)

        if img_file is not None:
            # load the image and convert it to grayscale
            image = load_image_PIL(img_file)
            image = converted(image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # construct a rectangular kernel (13x5) and apply a blackhat operation which enables
            # us to find dark regions on a light background
            rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))
            blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKernel)

            # similarly, a tophat (also called a "whitehat") operation will enable us to find light
            # regions on a dark background
            tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)

            st.subheader('Morphological Hats')
            # show the output images
            with st.expander('Show Original Image'):
                st.markdown("Original")
                st.image(image)
            with st.expander('Show Morphological Hats', expanded=True):
                cols = st.columns(2)
                cols[0].markdown("Blackhat")
                cols[0].image(blackhat)
                with cols[0]:
                    download_button1(blackhat, button,
                                     download, mime_type, key='1.1')
                cols[1].markdown("Tophat")
                cols[1].image(tophat)
                with cols[1]:
                    download_button1(tophat, button, download,
                                     mime_type, key='1.2')
        else:
            # load the image and convert it to grayscale
            image = load_image('images/pyimagesearch_logo_noise.png')
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # construct a rectangular kernel (13x5) and apply a blackhat operation which enables
            # us to find dark regions on a light background
            rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))
            blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKernel)

            # similarly, a tophat (also called a "whitehat") operation will enable us to find light
            # regions on a dark background
            tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)

            st.subheader('Morphological Hats')
            # show the output images
            with st.expander('Show Original Image'):
                st.markdown("Original")
                st.image(image)
            with st.expander('Show Morphological Hats', expanded=True):
                cols = st.columns(2)
                cols[0].markdown("Blackhat")
                cols[0].image(blackhat)
                with cols[0]:
                    download_button1(blackhat, button,
                                     download, mime_type, key='1.1')
                cols[1].markdown("Tophat")
                cols[1].image(tophat)
                with cols[1]:
                    download_button1(tophat, button, download,
                                     mime_type, key='1.2')

    else:
        img_file = st.file_uploader(
            label='Upload a file', type=['png', 'jpg', 'jpge'])
        realtime_update = st.sidebar.checkbox(
            label="Update in Real Time", value=True)
        if img_file is not None:
            # load the image, convert it to grayscale, and display it to our screen
            image = load_image_PIL(img_file)
            image = converted(image)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            with st.expander('Show Morphological Operations - Erosion', expanded=True):
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)
                # apply a series of erosions
                for i in range(0, 3):
                    eroded = cv.erode(gray.copy(), None, iterations=i + 1)
                    cols[i+1].markdown("Eroded {} times".format(i + 1))
                    cols[i+1].image(eroded)
                    with cols[i+1]:
                        download_button1(eroded, button, download,
                                         mime_type, key=f'{i}.{i}')

            with st.expander('Show Morphological Operations - Dilation'):
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)
                # apply a series of dilations
                for i in range(0, 3):
                    dilated = cv.dilate(gray.copy(), None, iterations=i + 1)
                    cols[i+1].markdown("Dilated {} times".format(i + 1))
                    cols[i+1].image(dilated)
                    with cols[i+1]:
                        download_button1(
                            dilated, button, download, mime_type, key=f'{i}.{i}')

            with st.expander('Show Morphological Operations - Opening'):
                # initialize a list of kernels sizes that will be applied to the image
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)
                kernelSizes = [(3, 3), (5, 5), (7, 7)]

                # loop over the kernels sizes
                for i, kernelSize in enumerate(kernelSizes):
                    # construct a rectangular kernel from the current size and then
                    # apply an "opening" operation
                    kernel = cv.getStructuringElement(
                        cv.MORPH_RECT, kernelSize)
                    opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
                    cols[i+1].markdown("Opening: ({}, {})".format(
                        kernelSize[0], kernelSize[1]))
                    cols[i+1].image(opening)
                    with cols[i+1]:
                        download_button1(
                            opening, button, download, mime_type, key=f'{i}.{i}')

            with st.expander('Show Morphological Operations - Closing'):
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)

                # loop over the kernels sizes again
                for i, kernelSize in enumerate(kernelSizes):
                    # construct a rectangular kernel form the current size, but this
                    # time apply a "closing" operation
                    kernel = cv.getStructuringElement(
                        cv.MORPH_RECT, kernelSize)
                    closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
                    cols[i+1].markdown("Closing: ({}, {})".format(
                        kernelSize[0], kernelSize[1]))
                    cols[i+1].image(closing)
                    with cols[i+1]:
                        download_button1(
                            closing, button, download, mime_type, key=f'{i}.{i}')

            with st.expander('Show Morphological Operations - Gradient'):
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)

                # loop over the kernels a final time
                for i, kernelSize in enumerate(kernelSizes):
                    # construct a rectangular kernel and apply a "morphological
                    # gradient" operation to the image
                    kernel = cv.getStructuringElement(
                        cv.MORPH_RECT, kernelSize)
                    gradient = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
                    cols[i+1].markdown("Gradient: ({}, {})".format(
                        kernelSize[0], kernelSize[1]))
                    cols[i+1].image(gradient)
                    with cols[i+1]:
                        download_button1(gradient, button,
                                         download, mime_type, key=f'{i}.{i}')

            with st.expander('Show Interactive Morphological Operations - Erosion, Dilation', expanded=True):
                x = st.number_input('Erored-Dilated Iterations', 1, 6)
                cols = st.columns(3)
                cols[0].markdown("Original")
                cols[0].image(image)
                eroded = cv.erode(gray.copy(), None, iterations=x)
                cols[1].markdown("Eroded {} times".format(x))
                cols[1].image(eroded)
                with cols[1]:
                    download_button1(eroded, button, download,
                                     mime_type, key='4.1')

                dilated = cv.dilate(gray.copy(), None, iterations=x)
                cols[2].markdown("Dilated {} times".format(x))
                cols[2].image(dilated)
                with cols[2]:
                    download_button1(dilated, button, download,
                                     mime_type, key='4.2')

            with st.expander('Show Interactive Morphological Operations - Opening, Closing & Gradient'):
                kX = st.number_input(
                    'Opening, Closing & Gradient Kernel Size', 1, 11, step=2)
                kY = st.number_input('Opening, Closing & Gradient Kernel Size', int(
                    kX), 11, step=2, disabled=True)
                kernelSize = [(kX, kY)]
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)

                kernel = cv.getStructuringElement(cv.MORPH_RECT, (kX, kY))
                opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
                cols[1].markdown("Opening: ({}, {})".format(
                    kernelSize[0][0], kernelSize[0][1]))
                cols[1].image(opening)
                with cols[1]:
                    download_button1(eroded, button, download,
                                     mime_type, key='5.1')

                kernel = cv.getStructuringElement(cv.MORPH_RECT, (kX, kY))
                closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
                cols[2].markdown("Closing: ({}, {})".format(
                    kernelSize[0][0], kernelSize[0][1]))
                cols[2].image(closing)
                with cols[2]:
                    download_button1(closing, button, download,
                                     mime_type, key='5.2')

                kernel = cv.getStructuringElement(cv.MORPH_RECT, (kX, kY))
                gradient = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
                cols[3].markdown("Gradient: ({}, {})".format(
                    kernelSize[0][0], kernelSize[0][1]))
                cols[3].image(gradient)
                with cols[3]:
                    download_button1(gradient, button,
                                     download, mime_type, key='5.3')
        else:
            # load the image, convert it to grayscale, and display it to our screen
            image = load_image('images/pyimagesearch_logo_noise.png')
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            with st.expander('Show Morphological Operations - Erosion', expanded=True):
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)
                # apply a series of erosions
                for i in range(0, 3):
                    eroded = cv.erode(gray.copy(), None, iterations=i + 1)
                    cols[i+1].markdown("Eroded {} times".format(i + 1))
                    cols[i+1].image(eroded)
                    with cols[i+1]:
                        download_button1(eroded, button, download,
                                         mime_type, key=f'{i}.{i}')

            with st.expander('Show Morphological Operations - Dilation'):
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)
                # apply a series of dilations
                for i in range(0, 3):
                    dilated = cv.dilate(gray.copy(), None, iterations=i + 1)
                    cols[i+1].markdown("Dilated {} times".format(i + 1))
                    cols[i+1].image(dilated)
                    with cols[i+1]:
                        download_button1(
                            dilated, button, download, mime_type, key=f'{i}.{i}')

            with st.expander('Show Morphological Operations - Opening'):
                # initialize a list of kernels sizes that will be applied to the image
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)
                kernelSizes = [(3, 3), (5, 5), (7, 7)]

                # loop over the kernels sizes
                for i, kernelSize in enumerate(kernelSizes):
                    # construct a rectangular kernel from the current size and then
                    # apply an "opening" operation
                    kernel = cv.getStructuringElement(
                        cv.MORPH_RECT, kernelSize)
                    opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
                    cols[i+1].markdown("Opening: ({}, {})".format(
                        kernelSize[0], kernelSize[1]))
                    cols[i+1].image(opening)
                    with cols[i+1]:
                        download_button1(
                            opening, button, download, mime_type, key=f'{i}.{i}')

            with st.expander('Show Morphological Operations - Closing'):
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)

                # loop over the kernels sizes again
                for i, kernelSize in enumerate(kernelSizes):
                    # construct a rectangular kernel form the current size, but this
                    # time apply a "closing" operation
                    kernel = cv.getStructuringElement(
                        cv.MORPH_RECT, kernelSize)
                    closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
                    cols[i+1].markdown("Closing: ({}, {})".format(
                        kernelSize[0], kernelSize[1]))
                    cols[i+1].image(closing)
                    with cols[i+1]:
                        download_button1(
                            closing, button, download, mime_type, key=f'{i}.{i}')

            with st.expander('Show Morphological Operations - Gradient'):
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)

                # loop over the kernels a final time
                for i, kernelSize in enumerate(kernelSizes):
                    # construct a rectangular kernel and apply a "morphological
                    # gradient" operation to the image
                    kernel = cv.getStructuringElement(
                        cv.MORPH_RECT, kernelSize)
                    gradient = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
                    cols[i+1].markdown("Gradient: ({}, {})".format(
                        kernelSize[0], kernelSize[1]))
                    cols[i+1].image(gradient)
                    with cols[i+1]:
                        download_button1(gradient, button,
                                         download, mime_type, key=f'{i}.{i}')

            with st.expander('Show Interactive Morphological Operations - Erosion, Dilation', expanded=True):
                x = st.number_input('Erored-Dilated Iterations', 1, 6)
                cols = st.columns(3)
                cols[0].markdown("Original")
                cols[0].image(image)
                eroded = cv.erode(gray.copy(), None, iterations=x)
                cols[1].markdown("Eroded {} times".format(x))
                cols[1].image(eroded)
                with cols[1]:
                    download_button1(eroded, button, download,
                                     mime_type, key='6.1')

                dilated = cv.dilate(gray.copy(), None, iterations=x)
                cols[2].markdown("Dilated {} times".format(x))
                cols[2].image(dilated)
                with cols[2]:
                    download_button1(dilated, button, download,
                                     mime_type, key='6.2')

            with st.expander('Show Interactive Morphological Operations - Opening, Closing & Gradient'):
                kX = st.number_input(
                    'Opening, Closing & Gradient Kernel Size', 1, 11, step=2)
                kY = st.number_input(
                    'Opening, Closing & Gradient Kernel Size', kX, 11, step=2, disabled=True)
                kernelSize = [(kX, kY)]
                cols = st.columns(4)
                cols[0].markdown("Original")
                cols[0].image(image)

                kernel = cv.getStructuringElement(cv.MORPH_RECT, (kX, kY))
                opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
                cols[1].markdown("Opening: ({}, {})".format(
                    kernelSize[0][0], kernelSize[0][1]))
                cols[1].image(opening)
                with cols[1]:
                    download_button1(eroded, button, download,
                                     mime_type, key='7.1')

                kernel = cv.getStructuringElement(cv.MORPH_RECT, (kX, kY))
                closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
                cols[2].markdown("Closing: ({}, {})".format(
                    kernelSize[0][0], kernelSize[0][1]))
                cols[2].image(closing)
                with cols[2]:
                    download_button1(closing, button, download,
                                     mime_type, key='7.2')

                kernel = cv.getStructuringElement(cv.MORPH_RECT, (kX, kY))
                gradient = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
                cols[3].markdown("Gradient: ({}, {})".format(
                    kernelSize[0][0], kernelSize[0][1]))
                cols[3].image(gradient)
                with cols[3]:
                    download_button1(gradient, button,
                                     download, mime_type, key='7.3')

    source_code(
        'Source Code + Morphological Operations Tutorial pyimagesearch.com',
        'https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/',
        'https://gist.github.com/jjaramillo34/3c1a8489e7882a3dba1127f3046c2a78')

    with st.expander('DuckDuckGo Search Results'):
        st.subheader('More About Morphological Operations')
        scrape_duckduckgo('morphological operations opencv')


def thresholding():
    st.header("Thresholding Demo")
    options = st.sidebar.radio(
        'Thresholding Options', ('Simple Thresholding', "Otsu's Thresholding"))
    realtime_update = st.sidebar.checkbox(
        label="Update in Real Time", value=True)

    if options == 'Simple Thresholding':
        img_file = st.file_uploader(
            label='Upload a file', type=['png', 'jpg', 'jpge'])
        if img_file is not None:
            image = load_image_PIL(img_file)
            image = converted(image)
            with st.expander('Show Original Image'):
                st.markdown(original)
                st.image(image)

            # convert the image to grayscale and blur it slightly
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (7, 7), 0)

            with st.expander('Show Simple Thresholding', expanded=True):
                cols = st.columns(3)
                # apply basic thresholding -- the first parameter is the image we want to thhreshold, the second value
                # is our threshold check; if a pixel value is greater than out threshold (in this case, 200), we set
                # it to be *black, otherwise it is *white*
                (T, threshInv) = cv.threshold(
                    blurred, 200, 255, cv.THRESH_BINARY_INV)
                cols[0].markdown("Threshold Binary Inverse")
                cols[0].image(threshInv)
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.1')

                # using normal thresholding (rather than inverse thresholding)
                (T, thresh) = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)
                cols[1].markdown("Threshold Binary")
                cols[1].image(thresh)
                with cols[1]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.2')

                # visualize only the masted regions in the image
                masked = cv.bitwise_and(image, image, mask=threshInv)
                cols[2].markdown("Masked")
                cols[2].image(masked)
                with cols[2]:
                    download_button1(masked, button, download,
                                     mime_type, key='1.3')

            with st.expander('Show Simple Thresholding Auto', expanded=True):
                x = st.slider('Change Threshold value',
                              min_value=50, max_value=255)
                cols = st.columns(3)
                # apply basic thresholding -- the first parameter is the image we want to thhreshold, the second value
                # is our threshold check; if a pixel value is greater than out threshold (in this case, 200), we set
                # it to be *black, otherwise it is *white*
                (T, threshInv) = cv.threshold(
                    blurred, x, 255, cv.THRESH_BINARY_INV)
                cols[0].markdown("Threshold Binary Inverse")
                cols[0].image(threshInv)
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='2.1')

                # using normal thresholding (rather than inverse thresholding)
                (T, thresh) = cv.threshold(blurred, x, 255, cv.THRESH_BINARY)
                cols[1].markdown("Threshold Binary")
                cols[1].image(thresh)
                with cols[1]:
                    download_button1(thresh, button, download,
                                     mime_type, key='2.2')

                # visualize only the masted regions in the image
                masked = cv.bitwise_and(image, image, mask=threshInv)
                cols[2].markdown("Masked")
                cols[2].image(masked)
                with cols[2]:
                    download_button1(masked, button, download,
                                     mime_type, key='2.3')
        else:
            image = load_image(default_image)
            with st.expander('Show Original Image'):
                st.markdown(original)
                st.image(image)

            # convert the image to grayscale and blur it slightly
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (7, 7), 0)

            with st.expander('Show Simple Thresholding', expanded=True):
                cols = st.columns(3)
                # apply basic thresholding -- the first parameter is the image we want to thhreshold, the second value
                # is our threshold check; if a pixel value is greater than out threshold (in this case, 200), we set
                # it to be *black, otherwise it is *white*
                (T, threshInv) = cv.threshold(
                    blurred, 200, 255, cv.THRESH_BINARY_INV)
                cols[0].markdown("Threshold Binary Inverse")
                cols[0].image(threshInv)
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.1')

                # using normal thresholding (rather than inverse thresholding)
                (T, thresh) = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)
                cols[1].markdown("Threshold Binary")
                cols[1].image(thresh)
                with cols[1]:
                    download_button1(thresh, button, download,
                                     mime_type, key='1.2')

                # visualize only the masted regions in the image
                masked = cv.bitwise_and(image, image, mask=threshInv)
                cols[2].markdown("Masked")
                cols[2].image(masked)
                with cols[2]:
                    download_button1(masked, button, download,
                                     mime_type, key='1.3')

            with st.expander('Show Simple Thresholding Auto', expanded=True):
                x = st.slider('Change Threshold value',
                              min_value=50, max_value=255)
                cols = st.columns(3)
                # apply basic thresholding -- the first parameter is the image we want to thhreshold, the second value
                # is our threshold check; if a pixel value is greater than out threshold (in this case, 200), we set
                # it to be *black, otherwise it is *white*
                (T, threshInv) = cv.threshold(
                    blurred, x, 255, cv.THRESH_BINARY_INV)
                cols[0].markdown("Threshold Binary Inverse")
                cols[0].image(threshInv)
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='2.1')

                # using normal thresholding (rather than inverse thresholding)
                (T, thresh) = cv.threshold(blurred, x, 255, cv.THRESH_BINARY)
                cols[1].markdown("Threshold Binary")
                cols[1].image(thresh)
                with cols[1]:
                    download_button1(thresh, button, download,
                                     mime_type, key='2.2')

                # visualize only the masted regions in the image
                masked = cv.bitwise_and(image, image, mask=threshInv)
                cols[2].markdown("Masked")
                cols[2].image(masked)
                with cols[2]:
                    download_button1(masked, button, download,
                                     mime_type, key='2.3')

    else:
        img_file = st.file_uploader(
            label='Upload a file', type=['png', 'jpg', 'jpge'])

        if img_file is not None:
            # load the image and display it
            image = load_image_PIL(img_file)
            image = converted(image)
            with st.expander('Show Original Image'):
                st.markdown("Image")
                st.image(image)

            # convert the image to grayscale and blur it slightly
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (7, 7), 0)

            with st.expander("Show Otsu's Thresholding", expanded=True):
                cols = st.columns(2)
                # apply Otsu's automatic thresholding which automatically determines
                # the best threshold value
                (T, threshInv) = cv.threshold(blurred, 0, 255,
                                              cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
                cols[0].markdown("Threshold")
                cols[0].image(threshInv)
                st.success("[INFO] otsu's thresholding value: {}".format(T))
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.1')

                # visualize only the masked regions in the image
                masked = cv.bitwise_and(image, image, mask=threshInv)
                cols[1].markdown("Output")
                cols[1].image(masked)
                with cols[1]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='1.2')

        else:
            # load the image and display it
            image = load_image(default_image)
            with st.expander('Show Original Image'):
                st.markdown("Image")
                st.image(image)

            # convert the image to grayscale and blur it slightly
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (7, 7), 0)

            with st.expander("Show Otsu's Thresholding", expanded=True):
                cols = st.columns(2)
                # apply Otsu's automatic thresholding which automatically determines
                # the best threshold value
                (T, threshInv) = cv.threshold(blurred, 0, 255,
                                              cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
                cols[0].markdown("Threshold")
                cols[0].image(threshInv)
                st.success("[INFO] otsu's thresholding value: {}".format(T))
                with cols[0]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='2.1')

                # visualize only the masked regions in the image
                masked = cv.bitwise_and(image, image, mask=threshInv)
                cols[1].markdown("Output")
                cols[1].image(masked)
                with cols[1]:
                    download_button1(threshInv, button,
                                     download, mime_type, key='2.2')

    source_code(
        'Source Code + Thresholding Tutorial pyimagesearch.com',
        'https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/',
        'https://gist.github.com/jjaramillo34/d504d5a9d6f88833c3720f132e734193')

    with st.expander('DuckDuckGo Search Results'):
        st.subheader('More About Thesholding')
        scrape_duckduckgo('opencv thresholding')


def color_spaces():
    st.header("Color Spaces Demo")
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpge'])
    realtime_update = st.sidebar.checkbox(
        label="Update in Real Time", value=True)

    if img_file is not None:
        with st.expander('Show RGB Color Spaces', expanded=True):
            # load the original image and show it
            cols = st.columns(4)
            image = load_image_PIL(img_file)
            image = converted(image)
            with cols[0]:
                st.markdown("RGB Color Spaces")
                st.image(image)

            # loop over each of the individual channels and display them
            for i, (name, chan) in enumerate(zip(("B", "G", "R"), cv.split(image))):
                with cols[i+1]:
                    st.markdown(name)
                    st.image(chan)
                    download_button1(chan, button, download,
                                     mime_type, key=f'{i}.{i}')

        with st.expander('Show HSV Color Spaces'):
            # convert the image to the HSV color space and show it
            cols = st.columns(4)
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            with cols[0]:
                st.markdown("HSV Color Spaces")
                st.image(hsv)

            # loop over each of the invidiaul channels and display them
            for i, (name, chan) in enumerate(zip(("H", "S", "V"), cv.split(hsv))):
                with cols[i+1]:
                    st.markdown(name)
                    st.image(chan)
                    download_button1(chan, button, download,
                                     mime_type, key=f'{i}.{i}')

        with st.expander('Show L*a*b* Color Spaces'):
            # convert the image to the L*a*b* color space and show it
            cols = st.columns(4)
            lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
            with cols[0]:
                st.markdown("L*a*b*")
                st.image(lab)

            # loop over each of the invidiaul channels and display them
            for i, (name, chan) in enumerate(zip(("L*", "a*", "b*"), cv.split(lab))):
                with cols[i+1]:
                    st.markdown(name)
                    st.image(chan)
                    download_button1(chan, button, download,
                                     mime_type, key=f'{i}.{i}')

        with st.expander('Show Grayscale'):
            # show the original and grayscale versions of the image
            cols = st.columns(2)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            cols[0].markdown("Original")
            cols[0].image(image)
            with cols[0]:
                download_button1(image, button, download,
                                 mime_type, key=f'2.1')
            cols[1].markdown("Grayscale")
            cols[1].image(gray)
            with cols[1]:
                download_button1(image, button, download,
                                 mime_type, key=f'2.2')
    else:
        with st.expander('Show RGB Color Spaces', expanded=True):
            # load the original image and show it
            cols = st.columns(4)
            image = load_image(default_image)
            with cols[0]:
                st.markdown("RGB Color Spaces")
                st.image(image)

            # loop over each of the individual channels and display them
            for i, (name, chan) in enumerate(zip(("B", "G", "R"), cv.split(image))):
                with cols[i+1]:
                    st.markdown(name)
                    st.image(chan)
                    download_button1(chan, button, download,
                                     mime_type, key=f'{i}.{i}')

        with st.expander('Show HSV Color Spaces'):
            # convert the image to the HSV color space and show it
            cols = st.columns(4)
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            with cols[0]:
                st.markdown("HSV Color Spaces")
                st.image(hsv)

            # loop over each of the invidiaul channels and display them
            for i, (name, chan) in enumerate(zip(("H", "S", "V"), cv.split(hsv))):
                with cols[i+1]:
                    st.markdown(name)
                    st.image(chan)
                    download_button1(chan, button, download,
                                     mime_type, key=f'{i}.{i}')

        with st.expander('Show L*a*b* Color Spaces'):
            # convert the image to the L*a*b* color space and show it
            cols = st.columns(4)
            lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
            with cols[0]:
                st.markdown("L*a*b*")
                st.image(lab)

            # loop over each of the invidiaul channels and display them
            for i, (name, chan) in enumerate(zip(("L*", "a*", "b*"), cv.split(lab))):
                with cols[i+1]:
                    st.markdown(name)
                    st.image(chan)
                    download_button1(chan, button, download,
                                     mime_type, key=f'{i}.{i}')

        with st.expander('Show Grayscale'):
            # show the original and grayscale versions of the image
            cols = st.columns(2)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            cols[0].markdown("Original")
            cols[0].image(image)
            with cols[0]:
                download_button1(image, button, download,
                                 mime_type, key=f'2.1')
            cols[1].markdown("Grayscale")
            cols[1].image(gray)
            with cols[1]:
                download_button1(image, button, download,
                                 mime_type, key=f'2.2')

    source_code(
        'Source Code + Color Spaces Tutorial pyimagesearch.com',
        'https://pyimagesearch.com/2021/04/28/opencv-color-spaces-cv2-cvtcolor/',
        'https://gist.github.com/jjaramillo34/74ef1a86014fb4fd7617c03ea10c3602')

    with st.expander('DuckDuckGo Search Results'):
        t = 'Color Spaces'
        st.subheader(f'More About {t.capitalize()}')
        scrape_duckduckgo(f'opencv {t}')


def smoothing_blurring():
    st.header("Smoothing & Blurring Demo")

    options = st.sidebar.radio(
        'Smoothing & Blurring Options', ('Bilateral', 'Blurring'))
    if options == 'Bilateral':

        img_file = st.file_uploader(
            label='Upload a file', type=['png', 'jpg', 'jpge'])
        realtime_update = st.sidebar.checkbox(
            label="Update in Real Time", value=True)

        if img_file is not None:
            with st.expander('Show Original Image'):
                image = load_image_PIL(img_file)
                image = converted(image)
                st.image(image)
            # hard-code parameters list
            params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]

            with st.expander('Bilateral Blurring', expanded=True):
                st.subheader('Bilateral Blurring')
                cols = st.columns(3)
                # loop over the diameter, sigma color, and sigma space
                for i, (diameter, sigmaColor, sigmaSpace) in enumerate(params):
                    # apply bilateral filtering to the image using the current set of parameters
                    blurred = cv.bilateralFilter(
                        image, diameter, sigmaColor, sigmaSpace)

                    # show the output image and associated parameters
                    title = "Blurred d={}, sc={}, ss={}".format(
                        diameter, sigmaColor, sigmaSpace)
                    with cols[i]:
                        st.markdown(title)
                        st.image(blurred)
                        download_button1(
                            blurred, button, download, mime_type, key=f'{i}')

            with st.expander('Bilateral Blurring Interactive'):
                st.subheader('Bilateral Blurring Interactive')
                cols = st.columns(3)
                d = cols[0].slider('Select starting diameter',
                                   min_value=11, max_value=100, step=1, key='1')
                sc = cols[1].slider(
                    'Select starting sigmaColor', min_value=21, max_value=100, step=1, key='2')
                ss = cols[2].slider(
                    'Select starting sigmaSpace', min_value=7, max_value=100, step=1, key='3')

                blurred = cv.bilateralFilter(image, d, sc, ss)

                # show the output image and associated parameters
                title = "Blurred d={}, sc={}, ss={}".format(
                    d, sc, ss)
                st.markdown(title)
                st.image(blurred)
                download_button1(blurred, button, download,
                                 mime_type, key='1.1')
        else:
            with st.expander('Show Original Image'):
                image = load_image(default_image)
                st.image(image)
            # hard-code parameters list
            params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]

            with st.expander('Bilateral Blurring', expanded=True):
                st.subheader('Bilateral Blurring')
                cols = st.columns(3)
                # loop over the diameter, sigma color, and sigma space
                for i, (diameter, sigmaColor, sigmaSpace) in enumerate(params):
                    # apply bilateral filtering to the image using the current set of parameters
                    blurred = cv.bilateralFilter(
                        image, diameter, sigmaColor, sigmaSpace)

                    # show the output image and associated parameters
                    title = "Blurred d={}, sc={}, ss={}".format(
                        diameter, sigmaColor, sigmaSpace)
                    with cols[i]:
                        st.markdown(title)
                        st.image(blurred)
                        download_button1(
                            blurred, button, download, mime_type, key=f'{i}')

            with st.expander('Bilateral Blurring Interactive'):
                st.subheader('Bilateral Blurring Interactive')
                cols = st.columns(3)
                d = cols[0].slider('Select starting diameter',
                                   min_value=11, max_value=100, step=1, key='1')
                sc = cols[1].slider(
                    'Select starting sigmaColor', min_value=21, max_value=100, step=1, key='2')
                ss = cols[2].slider(
                    'Select starting sigmaSpace', min_value=7, max_value=100, step=1, key='3')

                blurred = cv.bilateralFilter(image, d, sc, ss)

                # show the output image and associated parameters
                title = "Blurred d={}, sc={}, ss={}".format(
                    d, sc, ss)
                st.markdown(title)
                st.image(blurred)
                download_button1(blurred, button, download,
                                 mime_type, key='1.1')
    else:
        img_file = st.file_uploader(
            label='Upload a file', type=['png', 'jpg', 'jpge'])
        realtime_update = st.sidebar.checkbox(
            label="Update in Real Time", value=True)

        if img_file is not None:
            # load the image, display it to our screen, and initialize a list of
            # kernel sizes (so we can evaluate the relationship between kernel
            # size and amount of blurring)
            image = load_image_PIL(img_file)
            image = converted(image)
            with st.expander('Show Original Image'):
                st.image(image)
            kernelSizes = [(3, 3), (9, 9), (15, 15)]
            with st.expander('Show Average Blur', expanded=True):
                cols = st.columns(3)
                # loop over the kernel sizes
                for i, (kX, kY) in enumerate(kernelSizes):
                    # apply an "average" blur to the image using the current kernel size
                    with cols[i]:
                        blurred = cv.blur(image, (kX, kY))
                        st.markdown("Average Blur ({}, {})".format(kX, kY))
                        st.image(blurred)
                        download_button1(
                            blurred, button, download, mime_type, key=f'{i}')

            with st.expander('Show Gaussian Blur'):
                cols = st.columns(3)
                # loop over the kernel sizes again
                for i, (kX, kY) in enumerate(kernelSizes):
                    # apply a "Gaussian" blur to the image
                    with cols[i]:
                        blurred = cv.GaussianBlur(image, (kX, kY), 0)
                        st.markdown("Gaussian Blur ({}, {})".format(kX, kY))
                        st.image(blurred)
                        download_button1(
                            blurred, button, download, mime_type, key=f'{i}')

            with st.expander('Show Median Blur'):
                cols = st.columns(3)
                # loop over the kernel sizes a final time
                for i, k in enumerate((3, 9, 15)):
                    # apply a "median" blur to the image
                    with cols[i]:
                        blurred = cv.medianBlur(image, k)
                        st.markdown("Median Blur {}".format(k))
                        st.image(blurred)
                        download_button1(
                            blurred, button, download, mime_type, key=f'{i}')

            with st.expander('Show Auto Blurring', expanded=True):
                cols = st.columns(3)
                kX = cols[0].number_input(
                    'Kernel Sizes kX', min_value=1, max_value=25, step=2, key='2.1')
                kY = cols[1].number_input(
                    'Kernel Sizes kY', min_value=1, max_value=25, step=2, key='2.2', value=kX, disabled=True)
                k = cols[2].number_input(
                    'Kernel Sizes k', min_value=1, max_value=25, step=2, key='2.3')
                # apply an "average" blur to the image using the current kernel size
                blurred = cv.blur(image, (kX, kX))
                cols[0].markdown("Average Blur ({}, {})".format(kX, kY))
                cols[0].image(blurred)
                with cols[0]:
                    download_button1(blurred, button, download,
                                     mime_type, key='3.1')
                # apply a "Gaussian" blur to the image
                blurred = cv.GaussianBlur(image, (kX, kY), 0)
                cols[1].markdown("Gaussian Blur ({}, {})".format(kX, kY))
                cols[1].image(blurred)
                with cols[1]:
                    download_button1(blurred, button, download,
                                     mime_type, key='3.2')
                # apply a "median" blur to the image
                blurred = cv.medianBlur(image, k)
                cols[2].markdown("Median Blur {}".format(k))
                cols[2].image(blurred)
                with cols[2]:
                    download_button1(blurred, button, download,
                                     mime_type, key='3.3')
        else:
            # load the image, display it to our screen, and initialize a list of
            # kernel sizes (so we can evaluate the relationship between kernel
            # size and amount of blurring)
            image = load_image(default_image)
            with st.expander('Show Original Image'):
                st.image(image)
            kernelSizes = [(3, 3), (9, 9), (15, 15)]
            with st.expander('Show Average Blur', expanded=True):
                cols = st.columns(3)
                # loop over the kernel sizes
                for i, (kX, kY) in enumerate(kernelSizes):
                    # apply an "average" blur to the image using the current kernel size
                    with cols[i]:
                        blurred = cv.blur(image, (kX, kY))
                        st.markdown("Average Blur ({}, {})".format(kX, kY))
                        st.image(blurred)
                        download_button1(
                            blurred, button, download, mime_type, key=f'{i}')

            with st.expander('Show Gaussian Blur'):
                cols = st.columns(3)
                # loop over the kernel sizes again
                for i, (kX, kY) in enumerate(kernelSizes):
                    # apply a "Gaussian" blur to the image
                    with cols[i]:
                        blurred = cv.GaussianBlur(image, (kX, kY), 0)
                        st.markdown("Gaussian Blur ({}, {})".format(kX, kY))
                        st.image(blurred)
                        download_button1(
                            blurred, button, download, mime_type, key=f'{i}')

            with st.expander('Show Median Blur'):
                cols = st.columns(3)
                # loop over the kernel sizes a final time
                for i, k in enumerate((3, 9, 15)):
                    # apply a "median" blur to the image
                    with cols[i]:
                        blurred = cv.medianBlur(image, k)
                        st.markdown("Median Blur {}".format(k))
                        st.image(blurred)
                        download_button1(
                            blurred, button, download, mime_type, key=f'{i}')

            with st.expander('Show Auto Blurring', expanded=True):
                cols = st.columns(3)
                kX = cols[0].number_input(
                    'Kernel Sizes kX', min_value=1, max_value=25, step=2, key='2.1')
                kY = cols[1].number_input(
                    'Kernel Sizes kY', min_value=1, max_value=25, step=2, key='2.2', value=kX, disabled=True)
                k = cols[2].number_input(
                    'Kernel Sizes k', min_value=1, max_value=25, step=2, key='2.3')
                # apply an "average" blur to the image using the current kernel size
                blurred = cv.blur(image, (kX, kX))
                cols[0].markdown("Average Blur ({}, {})".format(kX, kY))
                cols[0].image(blurred)
                with cols[0]:
                    download_button1(blurred, button, download,
                                     mime_type, key='4.1')
                # apply a "Gaussian" blur to the image
                blurred = cv.GaussianBlur(image, (kX, kY), 0)
                cols[1].markdown("Gaussian Blur ({}, {})".format(kX, kY))
                cols[1].image(blurred)
                with cols[1]:
                    download_button1(blurred, button, download,
                                     mime_type, key='4.2')
                # apply a "median" blur to the image
                blurred = cv.medianBlur(image, k)
                cols[2].markdown("Median Blur {}".format(k))
                cols[2].image(blurred)
                with cols[2]:
                    download_button1(blurred, button, download,
                                     mime_type, key='4.3')

    source_code(
        f'Source Code + Smoothing and Blurring Tutorial pyimagesearch.com',
        'https://pyimagesearch.com/2021/04/28/opencv-smoothing-and-blurring/',
        'https://gist.github.com/jjaramillo34/84863214120f9e6bcf49874670250ebb')

    with st.expander('DuckDuckGo Search Results'):
        t = 'blurring and smoothing'
        st.subheader(f'More About {t.capitalize()}')
        scrape_duckduckgo(f'opencv {t}')


app()
