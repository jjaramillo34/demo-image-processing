import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
import string
import random
import requests
import shutil
import imutils
import streamlit.components.v1 as components
from datetime import datetime
from streamlit_cropper import st_cropper
from webcolors import hex_to_name
from PIL import Image, ImageColor
from streamlit_drawable_canvas import st_canvas
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
    source_code,
    tutorial_page,
    version)

from pages.page1.utils import drawing

# st.set_page_config(layout="wide")

selected_boxes = (
    "Welcome",
    "Demo Drawing",
    "Demo Crop",
    "Demo Flipping",
    "Demo Masking",
    "Demo Resizing",
    "Demo Rotate",
    "Demo Split-Merge",
    "Demo Translate",
)

aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}

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
        "Chosse one of the following", selected_boxes)

    print(selected_box)

    if selected_box == "Welcome":
        welcome()
    if selected_box == "Demo Drawing":
        drawing()
    if selected_box == "Demo Crop":
        cropping()
    if selected_box == "Demo Flipping":
        flipping()
    if selected_box == "Demo Resizing":
        resizing()
    if selected_box == "Demo Rotate":
        rotating()
    if selected_box == "Demo Masking":
        masking()
    if selected_box == "Demo Split-Merge":
        split_merge()
    if selected_box == "Demo Translate":
        translating()


def welcome():
    st.markdown("<h1 style='text-align: center; color: blue;'>OpenCV Basics</h1>",
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


def cropping():
    cropper_options = st.sidebar.radio(
        "Cropping Options",
        ('Streamlit-Cropper', 'OpenCV Cropper'))
    st.sidebar.markdown('Streamlit is **_really_ cool**.')

    st.set_option('deprecation.showfileUploaderEncoding', False)

    if cropper_options == 'Streamlit-Cropper':
        # Upload an image and set some options for demo purposes
        st.header("Cropping Demo using Streamlit-Cropper")
        img_file = st.file_uploader(label='Upload a file', type=[
                                    'png', 'jpg', 'jpge'], key='1')

        if img_file is not None:

            realtime_update = st.sidebar.checkbox(
                label="Update in Real Time", value=True)
            box_color = st.sidebar.color_picker(
                label="Box Color", value='#0000FF')
            aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=[
                                             "1:1", "16:9", "4:3", "2:3", "Free"])
            # Streamlit version
            version()

            aspect_ratio = aspect_dict[aspect_choice]

            if img_file:
                img = Image.open(img_file)
                if not realtime_update:
                    st.write("Double click to save crop")
                # Get a cropped image from the frontend
                cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                         aspect_ratio=aspect_ratio)

                # Manipulate cropped image at will
                st.write("Preview")
                _ = cropped_img.thumbnail((400, 400))
                st.image(cropped_img)

                st.markdown(download_button(cropped_img, download,
                            button, True), unsafe_allow_html=True)
        else:

            st.write(original)
            img = Image.open(default_image)
            #img = converted(img)

            realtime_update = st.sidebar.checkbox(
                label="Update in Real Time", value=True)
            box_color = st.sidebar.color_picker(
                label="Box Color", value='#0000FF')
            aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=[
                                             "1:1", "16:9", "4:3", "2:3", "Free"])
            # Streamlit version
            st.sidebar.caption(f"Streamlit version `{st.__version__}`")

            aspect_ratio = aspect_dict[aspect_choice]

            if not realtime_update:
                st.write("Double click to save crop")
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                     aspect_ratio=aspect_ratio)

            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((400, 400))
            st.image(cropped_img)

            st.markdown(download_button(cropped_img, download,
                        button, True), unsafe_allow_html=True)

    else:
        st.header("Croppign Demo using OpenCV")
        image = cv.imread(default_image)
        h = image.shape[0]
        w = image.shape[1]
        realtime_update = st.sidebar.checkbox(
            label="Update in Real Time", value=True)
        image_file = st.file_uploader(label='Upload a file', type=[
                                      'png', 'jpg', 'jpge'], key='2')
        st.write(original)
        st.image(image)

        col1, col2 = st.columns(2)
        with col1:
            x = st.slider('Select starting point on x axis',
                          min_value=0, max_value=w)
            st.write('Point X: ', x)
            y = st.slider('Select starting point on y axis',
                          min_value=0, max_value=h)
            st.write('Point Y:', y)
        with col2:
            image_width = st.slider(
                'Select width of cropped image', min_value=0, max_value=w - x)
            st.write('Image Width :', image_width)
            image_height = st.slider(
                'Select height of cropped image', min_value=0, max_value=h - y)
            st.write('Image Height :', image_height)

        if x == 0 or y == 0 or image_width == 0 or image_height == 0:
            pass
        else:
            cropped_image = image[y:y+image_height,
                                  x:x+image_width]  # Cropping using Slicing
            st.write("Preview")
            st.image(cropped_image)

            result = Image.fromarray(cropped_image)
            st.markdown(download_button(result, download,
                        button, True), unsafe_allow_html=True)


def flipping():
    st.header("Flipping Demo")
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpeg'])
    realtime_update = st.sidebar.checkbox(
        label="Update in Real Time", value=True)

    if img_file is not None:
        image = load_image_PIL(img_file)
        image = converted(image)
        st.write(original)
        st.image(image)

        flipped_button = st.button("🔄", on_click=increment_counter)

        if 'count' not in st.session_state:
            st.session_state.count = 0
            st.markdown("[INFO] Original Image...")
            st.image(image)
        elif st.session_state.count == 1:
            # flip the image vertically
            flipped = cv.flip(image, 0)
            st.markdown("[INFO] flipping image vertically...")
            st.write("Clicks:", st.session_state.count)
            st.image(flipped)
            result = Image.fromarray(flipped)
            st.markdown(download_button(result, download,
                        button, True), unsafe_allow_html=True)
            st.markdown("Source Code")
            st.code('''
# flip the image vertically
image = cv.imread("your_image")
flipped = cv.flip(image, 1)
st.markdown("[INFO] flipping image vertically...")
st.image(flipped)
''', language=language)

        elif st.session_state.count == 2:
            # flip the image horizontally
            flipped = cv.flip(image, 1)
            st.markdown("[INFO] flipping image horizontally...")
            st.image(flipped)
            result = Image.fromarray(flipped)
            st.markdown(download_button(result, download,
                        button, True), unsafe_allow_html=True)
            st.markdown("Source Code")
            st.code('''
# flip the image horizontally
flipped = cv.flip(image, 1)
st.markdown("[INFO] flipping image horizontally...")
st.image(flipped)''', language=language)
        elif st.session_state.count == 3:
            # flip the image along both axes
            flipped = cv.flip(image, -1)
            st.markdown("[INFO] flipping image horizontally and vertically...")
            st.image(flipped)
            result = Image.fromarray(flipped)
            st.markdown(download_button(result, download,
                        button, True), unsafe_allow_html=True)
            st.markdown("Source Code")
            st.code('''
# flip the image along both axes
flipped = cv.flip(image, -1)
st.markdown("[INFO] flipping image horizontally and vertically...")
st.image(flipped)''', language=language)
        elif st.session_state.count == 4:
            st.session_state.count = 0
            st.markdown("[INFO] Original Image...")
            st.image(image)

    else:
        image = load_image(default_image)
        st.write(original)
        st.image(image)

        flipped_button = st.button("🔄", on_click=increment_counter)

        if 'count' not in st.session_state:
            st.session_state.count = 0
            st.markdown("[INFO] Original Image...")
            st.image(image)
        elif st.session_state.count == 1:
            # flip the image vertically
            flipped = cv.flip(image, 0)
            st.markdown("[INFO] flipping image vertically...")
            st.write("Clicks:", st.session_state.count)
            st.image(flipped)
            result = Image.fromarray(flipped)
            st.markdown(download_button(result, download,
                        button, True), unsafe_allow_html=True)
            st.markdown("Source Code")
            st.code('''
# flip the image vertically
image = cv.imread("your_image")
flipped = cv.flip(image, 1)
st.markdown("[INFO] flipping image vertically...")
st.image(flipped)
''', language=language)

        elif st.session_state.count == 2:
            # flip the image horizontally
            flipped = cv.flip(image, 1)
            st.markdown("[INFO] flipping image horizontally...")
            st.image(flipped)
            result = Image.fromarray(flipped)
            st.markdown(download_button(result, download,
                        button, True), unsafe_allow_html=True)
            st.markdown("Source Code")
            st.code('''
# flip the image horizontally
flipped = cv.flip(image, 1)
st.markdown("[INFO] flipping image horizontally...")
st.image(flipped)''', language=language)
        elif st.session_state.count == 3:
            # flip the image along both axes
            flipped = cv.flip(image, -1)
            st.markdown("[INFO] flipping image horizontally and vertically...")
            st.image(flipped)
            st.markdown("Source Code")
            st.code('''
# flip the image along both axes
flipped = cv.flip(image, -1)
st.markdown("[INFO] flipping image horizontally and vertically...")
st.image(flipped)''', language=language)
        elif st.session_state.count == 4:
            st.session_state.count = 0
            st.markdown("[INFO] Original Image...")
            st.image(image)


def masking():
    st.header("Masking Demo")
    masking_options = st.sidebar.radio(
        "Masking Options",
        ('Simple Masking', 'Black/White, Color Masking'))
    realtime_update = st.sidebar.checkbox(
        label="Update in Real Time", value=True)
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpge'])

    if masking_options == 'Simple Masking':

        if img_file is not None:
            image = load_image_PIL(img_file)
            image = converted(image)
            st.markdown(original)
            st.image(image)
            # a mask is the same size as our image, but has only two pixels values, 0 to 255 -- pixels with
            # value of 0 (background) are ignored in the original image while mask pixels with a value 255
            # (foreground) are allowed to be kept
            mask = np.zeros(image.shape[:2], dtype="uint8")
            col1, col2 = st.columns(2)
            with col1:
                topLeft = st.slider(
                    'Select a range of TopLeft Corner', 0, image.shape[1], (0, 200))
            with col2:
                topRight = st.slider(
                    'Select a range of TopRight Corner', 0, image.shape[2], (400, 850))
            #cv.putText(image, f'({topLeft[0]+ 10}, {topLeft[1]+ 10})', topLeft, font, 4,(255,255,255),2,cv.LINE_AA)
            cv.rectangle(mask, topLeft, topRight, 255, -1)
            masked = cv.bitwise_and(image, image, mask=mask)

            # show the output images
            #st.markdown('Rectangular Mask')
            # st.image(mask)
            st.markdown("Mask applied to Image")
            st.image(masked)
            st.markdown("Source Code")
            st.code('''
# a mask is the same size as our image, but has only two pixels values, 0 to 255
# -- pixels with value of 0 (background) are ignored in the original image while
# mask pixels with a value 255 (foreground) are allowed to be kept
mask = np.zeros(image.shape[:2], dtype="uint8")
cv.rectangle(mask, (0, 200), (400, 850), 255, -1)
masked = cv.bitwise_and(image, image, mask=mask)

#show the output images
#st.markdown('Rectangular Mask')
#st.image(mask)
st.markdown("Mask applied to Image")
st.image(masked)''', language=language)

            # now let's make a circular mask with a radius of 298 pixels and appy the mask again
            mask = np.zeros(image.shape[:2], dtype="uint8")
            col1, col2 = st.columns(2)
            with col1:
                coor = st.slider('Select a range of Coordinates',
                                 0, image.shape[1], (300, 500))
            with col2:
                r = st.slider('Select a range of Radius', 0, 800, 298)
            cv.circle(mask, coor, r, 255, -1)
            masked = cv.bitwise_and(image, image, mask=mask)

            # show the output images
            #st.markdown("Circular Mask")
            # st.image(mask)
            st.markdown("Mask Applied to image")
            st.image(masked)
            st.markdown('Source Code')
            st.code('''
# now let's make a circular mask with a radius of 298 pixels and appy the mask again
mask = np.zeros(image.shape[:2], dtype="uint8")
cv.circle(mask, (300, 500), 300, 255, -1)
masked = cv.bitwise_and(image, image, mask=mask)

# show the output images
#st.markdown("Circular Mask")
#st.image(mask)
st.markdown("Mask Applied to image") 
st.image(masked)''', language=language)

        else:

            image = load_image(default_image)
            st.markdown(original)
            st.image(image)
            # a mask is the same size as our image, but has only two pixels values, 0 to 255 -- pixels with
            # value of 0 (background) are ignored in the original image while mask pixels with a value 255
            # (foreground) are allowed to be kept
            mask = np.zeros(image.shape[:2], dtype="uint8")
            col1, col2 = st.columns(2)
            with col1:
                topLeft = st.slider(
                    'Select a range of TopLeft Corner', 0, image.shape[1], (0, 200))
            with col2:
                topRight = st.slider(
                    'Select a range of TopRight Corner', 0, image.shape[2], (400, 850))
            #cv.putText(image, f'({topLeft[0]+ 10}, {topLeft[1]+ 10})', topLeft, font, 4,(255,255,255),2,cv.LINE_AA)
            cv.rectangle(mask, topLeft, topRight, 255, -1)
            masked = cv.bitwise_and(image, image, mask=mask)

            # show the output images
            #st.markdown('Rectangular Mask')
            # st.image(mask)
            st.markdown("Mask applied to Image")
            st.image(masked)
            st.markdown("Source Code")
            st.code('''
# a mask is the same size as our image, but has only two pixels values, 0 to 255
# -- pixels with value of 0 (background) are ignored in the original image while
# mask pixels with a value 255 (foreground) are allowed to be kept
mask = np.zeros(image.shape[:2], dtype="uint8")
cv.rectangle(mask, (0, 200), (400, 850), 255, -1)
masked = cv.bitwise_and(image, image, mask=mask)

#show the output images
#st.markdown('Rectangular Mask')
#st.image(mask)
st.markdown("Mask applied to Image")
st.image(masked)''', language=language)

            # now let's make a circular mask with a radius of 298 pixels and appy the mask again
            mask = np.zeros(image.shape[:2], dtype="uint8")
            col1, col2 = st.columns(2)
            with col1:
                coor = st.slider('Select a range of Coordinates',
                                 0, image.shape[1], (300, 500))
            with col2:
                r = st.slider('Select a range of Radius', 0, 800, 298)
            cv.circle(mask, coor, r, 255, -1)
            masked = cv.bitwise_and(image, image, mask=mask)
            # show the output images
            st.markdown("Mask Applied to image")
            st.image(masked)
            st.markdown('Source Code')
            st.code('''
# now let's make a circular mask with a radius of 298 pixels and appy the mask again
mask = np.zeros(image.shape[:2], dtype="uint8")
cv.circle(mask, (300, 500), 300, 255, -1)
masked = cv.bitwise_and(image, image, mask=mask)

# show the output images
st.markdown("Mask Applied to image") 
st.image(masked)''', language=language)

    else:
        col1, col2 = st.columns(2)
        with col1:
            color = st.color_picker('Pick A Color', '#FFFF00')
        with col2:
            hsv_coverted = ImageColor.getcolor(color, "RGB")
            st.markdown(
                '<p style="font-size: 14px">Color Hex / Color Name </p>', unsafe_allow_html=True)
            st.write(f'{color} / {convert_rgb_to_names(hsv_coverted)}')

        print(hsv_coverted)
        u = np.uint8([[[0, 236, 236]]])
        l = np.uint8([[[0, 236, 236]]])
        # define range of blue color in HSV
        lower_yellow = np.array(cv.cvtColor(l, cv.COLOR_BGR2HSV))
        upper_yellow = np.array(cv.cvtColor(u, cv.COLOR_BGR2HSV))

        hsv_upper = np.uint8(
            [[[hsv_coverted[2], hsv_coverted[1], hsv_coverted[0]]]])
        hsv_color = cv.cvtColor(hsv_upper, cv.COLOR_BGR2HSV)

        hsv_lower = [hsv_color[0][0][0] - 10, 100, 100]

        print(f'lower bound hsv: {lower_yellow}')
        print(f'upper bound hsv: {upper_yellow}')

        col1, col2 = st.columns(2)
        with col1:
            HMin = st.slider('HMin', min_value=0,
                             max_value=179, value=20, key='1')
            SMin = st.slider('SMin', min_value=0,
                             max_value=255, value=70, key='2')
            VMin = st.slider('VMin', min_value=0,
                             max_value=255, value=100, key='3')
        with col2:
            HMax = st.slider('HMax', min_value=0, max_value=179,
                             value=int(hsv_color[0][0][0]), key='4')
            SMax = st.slider('SMax', min_value=0, max_value=255,
                             value=int(hsv_color[0][0][1]), key='5')
            VMax = st.slider('VMax', min_value=0, max_value=255,
                             value=int(hsv_color[0][0][2]), key='6')
        #hMin = sMin = vMin = hMax = sMax = vMax = 0
        #phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        # Set minimum and maximum HSV values to display
        lower = np.array([HMin, SMin, VMin])
        upper = np.array([HMax, SMax, VMax])

        image = load_image('images/stop.jpg')
        st.markdown(original)
        st.image(image)
        # The kernel to be used for dilation purpose
        kernel = np.ones((5, 5), np.uint8)

        # converting the image to HSV format
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # defining the lower and upper values of HSV,
        # this will detect yellow colour
        Lower_hsv = np.array([20, 70, 100])
        Upper_hsv = np.array([30, 255, 255])

        # creating the mask by eroding,morphing,
        # dilating process
        Mask = cv.inRange(hsv, lower, upper)
        Mask = cv.erode(Mask, kernel, iterations=1)
        Mask = cv.morphologyEx(Mask, cv.MORPH_OPEN, kernel)
        Mask = cv.dilate(Mask, kernel, iterations=1)

        # Inverting the mask by
        # performing bitwise-not operation
        Mask = cv.bitwise_not(Mask)

        st.image(Mask)


def resizing():
    st.header("Resizing Demo")
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpge'])

    image = load_image('images/nice.jpeg')
    st.markdown('Original Image')
    st.image(image)

    options = st.sidebar.radio('Resize options', ('Resized (Width, Height), imutils',
                               'Interpolazion methods OpenCV', 'Resized (Auto) via imutils'))

    if options == 'Resized (Width, Height), imutils':
        # let's resize our image to be 300 pixels wide, but in order to
        # prevent our resized image from being skewed/distorted, we must
        # first calculate the ratio of the *new* width to the *old* width
        r = 300.0 / image.shape[1]
        dim = (300, int(image.shape[0] * r))

        # perform the actual resizing of the image
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        st.markdown("Resized (Width)")
        st.image(resized)
        st.markdown('Source Code')
        st.code(
            '''
# let's resize our image to be 300 pixels wide, but in order to
# prevent our resized image from being skewed/distorted, we must
# first calculate the ratio of the *new* width to the *old* width
r = 300.0 / image.shape[1]
dim = (300, int(image.shape[0] * r))

# perform the actual resizing of the image
resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
st.image(resized)''', language=language)

        # let's resize the image to have a height of 50 pixels, again keeping
        # in mind the aspect ratio
        r = 150.0 / image.shape[0]
        dim = (int(image.shape[1] * r), 150)

        # perform the resizing
        resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        st.markdown("Resized (Height)")
        st.image(resized)
        st.markdown('Source Code')
        st.code('''
# let's resize the image to have a height of 50 pixels, again keeping
# in mind the aspect ratio
r = 150.0 / image.shape[0]
dim = (int(image.shape[1] * r), 150)

# perform the resizing
resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
st.image(resized)

# calculating the ratio each and every time we want to resize an
# image is a real pain, so let's use the imutils convenience
# function which will *automatically* maintain our aspect ratio for us
resized = imutils.resize(image, width=400)
st.image(resized)''', language=language)

        # calculating the ratio each and every time we want to resize an
        # image is a real pain, so let's use the imutils convenience
        # function which will *automatically* maintain our aspect ratio for us
        resized = imutils.resize(image, width=400)
        st.markdown("Resized via imutils")
        st.image(resized)
        st.markdown('Source Code')
        st.code('''
resized = imutils.resize(image, width=400)
st.image(resized)''', language=language)

    elif options == 'Interpolazion methods OpenCV':
        # construct the list of interpolation methods in OpenCV
        methods = [
            ("cv.INTER_NEAREST", cv.INTER_NEAREST),
            ("cv.INTER_LINEAR", cv.INTER_LINEAR),
            ("cv.INTER_AREA", cv.INTER_AREA),
            ("cv.INTER_CUBIC", cv.INTER_CUBIC),
            ("cv.INTER_LANCZOS4", cv.INTER_LANCZOS4)]

        # loop over the interpolation methods
        for (name, method) in methods:
            # increase the size of the image by 3x using the current
            # interpolation method
            #print("[INFO] {}".format(name))
            resized = imutils.resize(image, width=image.shape[1] * 3,
                                     inter=method)
            st.markdown("Method: {}".format(name))
            st.image(resized)
        st.markdown('Source Code')
        st.code('''
# construct the list of interpolation methods in OpenCV
methods = [
    ("cv.INTER_NEAREST", cv.INTER_NEAREST),
    ("cv.INTER_LINEAR", cv.INTER_LINEAR),
    ("cv.INTER_AREA", cv.INTER_AREA),
    ("cv.INTER_CUBIC", cv.INTER_CUBIC),
    ("cv.INTER_LANCZOS4", cv.INTER_LANCZOS4)]

# loop over the interpolation methods
for (name, method) in methods:
    # increase the size of the image by 3x using the current
    # interpolation method
    resized = imutils.resize(image, width=image.shape[1] * 3,
        inter=method)
    st.image(resized)''', language=language)
    else:
        image_width = st.slider('Select width to Resized',
                                min_value=100, max_value=750, step=50)
        resized = imutils.resize(image, width=image_width)
        st.markdown("Resized via imutils with st.slider")
        st.image(resized)
        st.markdown('Source Code')
        st.code('''
image_width = st.slider('Select width to Resized', min_value = 100, max_value = 750, step=50)
resized = imutils.resize(image, width=image_width)
st.image(resized)''', language=language)
        st.markdown("Resized via imutils with st.number_input")
        image_width = st.number_input(
            'Insert pixels width', min_value=100, max_value=750, step=50)
        resized = imutils.resize(image, width=image_width)
        st.image(resized)
        st.markdown('Source Code')
        st.code('''
image_width = st.number_input('Insert pixels width', min_value = 100, max_value = 750, step=50)
resized = imutils.resize(image, width=image_width)
st.image(resized)''', language=language)


def rotating():
    st.header("Rotate Demo")
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpge'])
    image = load_image('images/nice.jpeg')

    options = st.sidebar.radio('Resize options', ('Rotate Cropping the Output Image',
                               'Rotate with imutils', 'Rotate with imutils Automatically '))

    if options == "Rotate Cropping the Output Image":

        # grab the dimensions of the image and calculate the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # rotate our image by 45 degrees around the center of the image
        M = cv.getRotationMatrix2D((cX, cY), 45, 1.0)
        rotated = cv.warpAffine(image, M, (w, h))
        st.markdown("Rotated by 45 degrees")
        st.image(rotated)

        # rotate our image by -90 degrees around the center of the image
        M = cv.getRotationMatrix2D((cX, cY), -90, 1.0)
        rotated = cv.warpAffine(image, M, (w, h))
        st.markdown("Rotated by 90 degrees")
        st.image(rotated)

        # rotate our image around an arbitrary point rather than the center
        M = cv.getRotationMatrix2D((10, 10), 45, 1.0)
        rotated = cv.warpAffine(image, M, (w, h))
        st.markdown("Rotated by 45 degrees"),
        st.image(rotated)
    elif options == 'Rotate with imutils':
        # use our imutils function to rotate an image 180 degrees
        rotated = imutils.rotate(image, 180)
        st.markdown("Rotated by 180 degrees")
        st.image(rotated)

        # rotate our image by 33 degrees counterclockwise, ensuring the
        # entire rotated image still views in the viewing area
        rotated = imutils.rotate_bound(image, -33)
        st.markdown("Rotated without Cropping")
        st.image(rotated)
    else:
        rotate_angle = st.slider(
            'Select width to Resized', min_value=-180, max_value=180, step=10)
        rotated = imutils.rotate_bound(image, rotate_angle)
        st.markdown("Resized via imutils with st.slider")
        st.image(rotated)
        st.markdown('Source Code')
        st.code('''
rotate_angle = st.slider('Select width to Resized', min_value = -180, max_value = 180, step=10)
rotated = imutils.rotate_bound(image, rotate_angle)
st.image(rotated)
''', language=language)
        st.markdown("Resized via imutils with st.number_input")
        rotate_angle = st.number_input(
            'Insert pixels width', min_value=-180, max_value=180, step=10)
        rotated = imutils.rotate_bound(image, rotate_angle)
        st.image(rotated)
        st.markdown('Source Code')
        st.code('''
rotate_angle = st.number_input('Insert pixels width', min_value = -180, max_value = 180, step=10)
rotated = imutils.rotate_bound(image, rotate_angle)
st.image(rotated)''', language=language)


def split_merge():
    st.header("Split-Merge Demo")
    image_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpge'])
    if image_file is not None:
        # load the input image and grab each channel -- note how OpenCV represents images as NumPy arrays
        # with channels in Blue, Green, Red ordering rather than Red, Green Bluest.markdown('Original')
        with st.expander('Show BGR Order Split-Merge OpenCV', expanded=True):
            cols = st.columns(4)
            image = load_image_PIL(image_file)
            image = converted(image)
            #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            cols[0].markdown('Original')
            cols[0].image(image)
            (B, G, R) = cv.split(image)
            # show each channel individually RGB order
            cols[1].markdown("Red")
            cols[1].image(R)
            with cols[1]:
                download_button1(R, button, download, mime_type, key="1.1")
            cols[2].markdown("Green")
            cols[2].image(G)
            with cols[2]:
                download_button1(G, button, download, mime_type, key="1.2")
            cols[3].markdown("Blue")
            cols[3].image(B)
            with cols[3]:
                download_button1(B, button, download, mime_type, key="1.3")

        with st.expander('Show RGB Order Split-Merge'):
            cols = st.columns(4)
            #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            cols[0].markdown('Original')
            cols[0].image(image)
            (R, G, B) = cv.split(image)
            # show each channel individually RGB order
            cols[1].markdown("Red")
            cols[1].image(B)
            with cols[1]:
                download_button1(B, button, download, mime_type, key="1.4")
            cols[2].markdown("Green")
            cols[2].image(G)
            with cols[2]:
                download_button1(G, button, download, mime_type, key="1.5")
            cols[3].markdown("Blue")
            cols[3].image(R)
            with cols[3]:
                download_button1(R, button, download, mime_type, key="1.6")

        with st.expander('Show BGR Order Color Split-Merge OpenCV', expanded=True):
            # merge the image back together again
            merged = cv.merge([B, G, R])
            merged = cv.cvtColor(merged, cv.COLOR_BGR2RGB)
            cols = st.columns(4)
            cols[0].markdown("Merged")
            cols[0].image(merged)
            # visualize each channel in color
            zeros = np.zeros(merged.shape[:2], dtype="uint8")
            cols[1].markdown("Red")
            cols[1].image(cv.merge([B, zeros, zeros]))
            with cols[1]:
                download_button1(
                    cv.merge([B, zeros, zeros]), button, download, mime_type, key="2.1")
            cols[2].markdown("Green")
            cols[2].image(cv.merge([zeros, G, zeros]))
            with cols[2]:
                download_button1(
                    cv.merge([zeros, G, zeros]), button, download, mime_type, key="2.2")
            cols[3].markdown("Blue")
            cols[3].image(cv.merge([zeros, zeros, R]))
            with cols[3]:
                download_button1(
                    cv.merge([zeros, zeros, R]), button, download, mime_type, key="2.3")

        with st.expander('Show RGB Order Color Split-Merge'):
            # merge the image back together again
            merged = cv.merge([B, G, R])
            merged = cv.cvtColor(merged, cv.COLOR_BGR2RGB)
            cols = st.columns(4)
            cols[0].markdown("Merged")
            cols[0].image(merged)
            # visualize each channel in color
            zeros = np.zeros(merged.shape[:2], dtype="uint8")
            cols[1].markdown("Red")
            cols[1].image(cv.merge([R, zeros, zeros]))
            with cols[1]:
                download_button1(
                    cv.merge([R, zeros, zeros]), button, download, mime_type, key="2.4")
            cols[2].markdown("Green")
            cols[2].image(cv.merge([zeros, G, zeros]))
            with cols[2]:
                download_button1(
                    cv.merge([zeros, G, zeros]), button, download, mime_type, key="2.5")
            cols[3].markdown("Blue")
            cols[3].image(cv.merge([zeros, zeros, B]))
            with cols[3]:
                download_button1(
                    cv.merge([zeros, zeros, B]), button, download, mime_type, key="2.6")

    else:
        st.header("Split-Merge Demo")
        # load the input image and grab each channel -- note how OpenCV represents images as NumPy arrays
        # with channels in Blue, Green, Red ordering rather than Red, Green Blue
        with st.expander('Show BGR Order Split-Merge OpenCV', expanded=True):
            cols = st.columns(4)
            image = load_image('images/RGB.jpg')
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            cols[0].markdown('Original')
            cols[0].image(image)
            (B, G, R) = cv.split(image)
            # show each channel individually RGB order
            cols[1].markdown("Red")
            cols[1].image(R)
            with cols[1]:
                download_button1(R, button, download, mime_type, key="1.1")
            cols[2].markdown("Green")
            cols[2].image(G)
            with cols[2]:
                download_button1(G, button, download, mime_type, key="1.2")
            cols[3].markdown("Blue")
            cols[3].image(B)
            with cols[3]:
                download_button1(B, button, download, mime_type, key="1.3")

        with st.expander('Show RGB Order Split-Merge'):
            cols = st.columns(4)
            image = load_image('images/RGB.jpg')
            #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            cols[0].markdown('Original')
            cols[0].image(image)
            (R, G, B) = cv.split(image)
            # show each channel individually RGB order
            cols[1].markdown("Red")
            cols[1].image(B)
            with cols[1]:
                download_button1(B, button, download, mime_type, key="1.4")
            cols[2].markdown("Green")
            cols[2].image(G)
            with cols[2]:
                download_button1(G, button, download, mime_type, key="1.5")
            cols[3].markdown("Blue")
            cols[3].image(R)
            with cols[3]:
                download_button1(R, button, download, mime_type, key="1.6")

        with st.expander('Show BGR Order Color Split-Merge OpenCV', expanded=True):
            # merge the image back together again
            merged = cv.merge([B, G, R])
            cols = st.columns(4)
            cols[0].markdown("Merged")
            cols[0].image(merged)
            # visualize each channel in color
            zeros = np.zeros(merged.shape[:2], dtype="uint8")
            st.write(merged.shape[:2])
            cols[1].markdown("Red")
            cols[1].image(cv.merge([B, zeros, zeros]))
            with cols[1]:
                download_button1(
                    cv.merge([B, zeros, zeros]), button, download, mime_type, key="2.1")
            cols[2].markdown("Green")
            cols[2].image(cv.merge([zeros, G, zeros]))
            with cols[2]:
                download_button1(
                    cv.merge([zeros, G, zeros]), button, download, mime_type, key="2.2")
            cols[3].markdown("Blue")
            cols[3].image(cv.merge([zeros, zeros, R]))
            with cols[3]:
                download_button1(
                    cv.merge([zeros, zeros, R]), button, download, mime_type, key="2.3")

        with st.expander('Show RGB Order Color Split-Merge'):
            # merge the image back together again
            merged = cv.merge([B, G, R])
            merged = cv.cvtColor(merged, cv.COLOR_BGR2RGB)
            cols = st.columns(4)
            cols[0].markdown("Merged")
            cols[0].image(merged)
            # visualize each channel in color
            zeros = np.zeros(merged.shape[:2], dtype="uint8")
            cols[1].markdown("Red")
            cols[1].image(cv.merge([R, zeros, zeros]))
            with cols[1]:
                download_button1(
                    cv.merge([R, zeros, zeros]), button, download, mime_type, key="2.4")
            cols[2].markdown("Green")
            cols[2].image(cv.merge([zeros, G, zeros]))
            with cols[2]:
                download_button1(
                    cv.merge([zeros, G, zeros]), button, download, mime_type, key="2.5")
            cols[3].markdown("Blue")
            cols[3].image(cv.merge([zeros, zeros, B]))
            with cols[3]:
                download_button1(
                    cv.merge([zeros, zeros, B]), button, download, mime_type, key="2.6")

    tutorial('Split-Merge pyimagesearch.com',
             'https://pyimagesearch.com/2021/01/23/splitting-and-merging-channels-with-opencv/')
    #gist_code('Source Code split_merge.py', 'https://pastebin.com/33pYnSHG')
    gitlab_code('Source Code split_merge.py',
                'https://gitlab.com/-/snippets/2296135')

    with st.expander('DuckDuckGo Search Results'):
        st.subheader('More About Split-Merge Thresholding')
        scrape_duckduckgo('split-merge opencv')


def translating():
    st.header("Translate Demo")
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpge'])

    if img_file is not None:

        with st.expander('Show Image Translate'):
            image = load_image_PIL(img_file)
            image = converted(image)
            cols = st.columns(3)
            # load the image and display it to our screen
            cols[0].markdown(original)
            cols[0].image(image)

            # shift the image 25 pixels to the right and 50 pixels down
            M = np.float32([[1, 0, 25], [0, 1, 50]])
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            cols[1].markdown("Shifted Down and Right")
            cols[1].image(shifted)
            with cols[1]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.1')

            # shift the image 25 pixels to the left and 50 pixels down
            M = np.float32([[1, 0, -50], [0, 1, 25]])
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            cols[2].markdown("Shifted Down and Left")
            cols[2].image(shifted)

        with cols[2]:
            download_button1(shifted, button, download, mime_type, key='1.2')

        with st.expander('Show Image Translate'):
            cols = st.columns(3)
            # now, let's shift the image 50 pixels to the left and 90 pixels
            # up by specifying negative values for the x and y directions respectively
            M = np.float32([[1, 0, -50], [0, 1, -90]])
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            cols[0].markdown("Shifted Up and and Left")
            cols[0].image(shifted)
            with cols[0]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.3')

            # now, let's shift the image 50 pixels to the right and 90 pixels
            # up by specifying negative values for the x and y directions respectively
            M = np.float32([[1, 0, 50], [0, 1, -90]])
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            cols[1].markdown("Shifted Up and and Right")
            cols[1].image(shifted)
            with cols[1]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.4')

            # use the imutils helper function to translate the image 100 pixels
            # down in a single function call
            shifted = imutils.translate(image, 0, 100)
            cols[2].markdown("Shifted Down")
            cols[2].image(shifted)
            with cols[2]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.5')
    else:
        with st.expander('Show Image Translate', expanded=True):
            cols = st.columns(3)
            # load the image and display it to our screen
            image = load_image(default_image)
            cols[0].markdown(original)
            cols[0].image(image)

            # shift the image 25 pixels to the right and 50 pixels down
            M = np.float32([[1, 0, 25], [0, 1, 50]])
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            cols[1].markdown("Shifted Down and Right")
            cols[1].image(shifted)
            with cols[1]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.1')

            # shift the image 25 pixels to the left and 50 pixels down
            M = np.float32([[1, 0, -50], [0, 1, 25]])
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            cols[2].markdown("Shifted Down and Left")
            cols[2].image(shifted)
            with cols[2]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.2')

        with st.expander('Show Image Translate'):
            cols = st.columns(3)
            # now, let's shift the image 50 pixels to the left and 90 pixels
            # up by specifying negative values for the x and y directions respectively
            M = np.float32([[1, 0, -50], [0, 1, -90]])
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            cols[0].markdown("Shifted Up and and Left")
            cols[0].image(shifted)
            with cols[0]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.3')

            # now, let's shift the image 50 pixels to the right and 90 pixels
            # up by specifying negative values for the x and y directions respectively
            M = np.float32([[1, 0, 50], [0, 1, -90]])
            shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
            cols[1].markdown("Shifted Up and and Right")
            cols[1].image(shifted)
            with cols[1]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.4')

            # use the imutils helper function to translate the image 100 pixels
            # down in a single function call
            shifted = imutils.translate(image, 0, 100)
            cols[2].markdown("Shifted Down")
            cols[2].image(shifted)
            with cols[2]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.5')

        with st.expander('Show Image Translate Interactive'):
            cols = st.columns(2)
            # use the imutils helper function to translate the image 100 pixels
            # down in a single function call
            #x = cols[0].number_input("X Value Left - Right", min_value=-2000, max_value=2000, value=0, step=100, key='1.1')
            x = cols[0].slider(
                'Select a range of values X Left - Right', -2000, 2000, 0, step=100, key='1.1')
            y = cols[0].number_input(
                "Y Value Up - Down", min_value=-2000, max_value=2000, value=0, step=100, key='1.2')
            shifted = imutils.translate(image, x, y)
            cols[0].markdown("Shifted Down")
            cols[0].image(shifted)
            with cols[0]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.6')

            shifted = imutils.translate(image, 0, 100)
            cols[1].markdown("Shifted Down")
            cols[1].image(shifted)
            with cols[1]:
                download_button1(shifted, button, download,
                                 mime_type, key='1.7')


app()
