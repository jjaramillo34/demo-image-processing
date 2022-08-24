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


def drawing():
    print(button)
    st.header("Drawing Demo")
    options = st.sidebar.radio(
        'Drawing Options', ('Drawing Basics', 'Drawing OpenCV Logo', 'Drawable Canvas - Streamlit'))
    if options == 'Drawing Basics':
        with st.expander('Show Drawing Basics', expanded=True):
            cols = st.columns(3)
            # initialize our canvas as a 500x500 pixel image with 3 channels (Red, Green, and Blue) with a black background
            canvas = np.zeros((500, 500, 3), dtype="uint8")

            # draw a green line from the top-left corner of our canvas to the bottom-right
            green = (0, 255, 0)
            cv.line(canvas, (0, 0), (500, 500), green)

            # draw a 3 pixel thick red line from the top-right corner to the bottom-left
            red = (0, 0, 255)
            cv.line(canvas, (500, 0), (0, 500), red, 3)

            # draw a green 50x50 pixel square, starting at 10x10 and ending at 60x60
            cv.rectangle(canvas, (10, 10), (60, 60), green)

            # draw another rectangle, this one red with 5 pixel thickness
            cv.rectangle(canvas, (50, 200), (200, 225), red, 5)

            # draw a final rectangle (blue and filled in )
            blue = (255, 0, 0)
            cv.rectangle(canvas, (200, 50), (225, 125), blue, -1)
            # display our image
            cols[0].markdown("Canvas")
            cols[0].image(canvas)
            with cols[0]:
                download_button1(canvas, button, download,
                                 mime_type, key="1.1")

            # re-initialize our canvas as an empty array, then compute the center (x, y)-coordinates of the canvas
            canvas = np.zeros((500, 500, 3), dtype="uint8")
            (centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
            white = (255, 255, 255)

            # loop over increasing radii, from 25 pixels to 150 pixels in 25 pixel increments
            for r in range(0, 525, 25):
                # draw a white circle with the current radius size
                cv.circle(canvas, (centerX, centerY), r, white)
            # display our image
            cols[1].markdown("Canvas")
            cols[1].image(canvas)
            with cols[1]:
                download_button1(canvas, button, download,
                                 mime_type, key="1.2")

            # re-initialize our canvas once again
            canvas = np.zeros((500, 500, 3), dtype="uint8")

            # let's draw 25 random circles
            for i in range(0, 25):
                # randomly generate a radius size between 5 and 200, generate a random color, and then pick a random
                # point on our canvas where the circle will be drawn
                radius = np.random.randint(5, high=200)
                color = np.random.randint(0, high=256, size=(3,)).tolist()
                pt = np.random.randint(0, high=500, size=(2,))

                # draw our random circle on the canvas
                cv.circle(canvas, tuple(pt), radius, color, -1)
            # display our image
            cols[2].markdown("Canvas")
            cols[2].image(canvas)
            with cols[2]:
                download_button1(canvas, button, download,
                                 mime_type, key="1.3")

        tutorial_page(
            'Drawing pyimagesearch.com',
            'https://pyimagesearch.com/2021/01/27/drawing-with-opencv/',)

        source_code(
            'Source Code + Drawing pyimagesearch.com',
            'https://gist.github.com/jjaramillo34/fb83acff62ce6502c398ba7133ab066c')

        with st.expander('DuckDuckGo Search Results'):
            st.subheader('More About Drawing')
            scrape_duckduckgo('basic drawing opencv')

    elif options == 'Drawing OpenCV Logo':

        with st.expander('Show Drawing OpenCV Logo', expanded=True):
            cols = st.columns(2)
            # initialize our canvas as a 300x300 pixel image with 3 channels (Red, Green, and Blue) with a black background
            image = np.full((360, 512, 3), 255, dtype="uint8")
            # draw ellipsis to recreate logo for OpenCV
            image = cv.ellipse(image, (256, 80), (60, 60),
                               120, 0, 300, (0, 0, 255), -1)
            image = cv.ellipse(image, (256, 80), (20, 20),
                               120, 0, 300, (255, 255, 255), -1)
            image = cv.ellipse(image, (176, 200), (60, 60),
                               0, 0, 300, (0, 255, 0), -1)
            image = cv.ellipse(image, (176, 200), (20, 20),
                               0, 0, 300, (255, 255, 255), -1)
            image = cv.ellipse(image, (336, 200), (60, 60),
                               300, 0, 300, (255, 0, 0), -1)
            image = cv.ellipse(image, (336, 200), (20, 20),
                               300, 0, 300, (255, 255, 255), -1)

            # draw OpenCV text
            image = cv.putText(image, "OpenCV", (196, 296),
                               font, 1, (0, 0, 0), 4, cv.LINE_AA)

            # Find Canny edges
            edged = cv.Canny(image, 30, 200)

            # Finding Contours
            # Use a copy of the image e.g. edged.copy() since findContours alters the image
            contours, hierarchy = cv.findContours(
                edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # Draw all contours -1 signifies drawing all contours
            cv.drawContours(image, contours, -1, (0, 0, 0), 2)

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv.filter2D(image, -1, kernel)

            # output draw image
            cols[0].markdown('Drawn OpenCV Original Logo')
            cols[0].image(image)
            with cols[0]:
                download_button1(image, button, download, mime_type, key="1.1")

            image = np.full((360, 512, 3), 255, dtype="uint8")

            # random color list
            color = [
                np.random.randint(0, high=256, size=(3,)).tolist(),
                np.random.randint(0, high=256, size=(3,)).tolist(),
                np.random.randint(0, high=256, size=(3,)).tolist()
            ]
            # draw ellipsis to recreate logo for OpenCV
            image = cv.ellipse(image, (256, 80), (60, 60),
                               120, 0, 300, color[0], -1)
            image = cv.ellipse(image, (256, 80), (20, 20),
                               120, 0, 300, (255, 255, 255), -1)
            image = cv.ellipse(image, (176, 200), (60, 60),
                               0, 0, 300, color[1], -1)
            image = cv.ellipse(image, (176, 200), (20, 20),
                               0, 0, 300, (255, 255, 255), -1)
            image = cv.ellipse(image, (336, 200), (60, 60),
                               300, 0, 300, color[2], -1)
            image = cv.ellipse(image, (336, 200), (20, 20),
                               300, 0, 300, (255, 255, 255), -1)

            # draw OpenCV text
            image = cv.putText(image, "OpenCV", (196, 296),
                               font, 1, (0, 0, 0), 4, cv.LINE_AA)

            # output draw image
            cols[1].markdown('Drawn OpenCV Random Colors Logo')
            cols[1].image(image)
            with cols[1]:
                download_button1(image, button, download,
                                 mime_type, key="drawing_1.1")

    else:
        # Specify canvas parameters in application
        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:", ("point", "freedraw", "line",
                              "rect", "circle", "polygon", "transform")
        )

        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider(
                "Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

        realtime_update = st.sidebar.checkbox("Update in realtime", True)
        bg_image = st.file_uploader("Background image:", type=["png", "jpg"])

        cols = st.columns(2)
        # Create a canvas component
        with cols[0]:
            canvas_result = st_canvas(
                # Fixed fill color with some opacity
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                #background_image=Image.open(bg_image) if bg_image else None,
                background_image=bg_image,
                update_streamlit=realtime_update,
                height=500,
                drawing_mode=drawing_mode,
                point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                key="canvas",
            )

        # Do something interesting with the image data and paths
        with cols[1]:
            if canvas_result.image_data is not None:
                if bg_image is not None:
                    #image = load_image_PIL(bg_image)
                    #image = converted(image)
                    # st.image(image)
                    st.image(canvas_result.image_data)
                    # print(canvas_result.image_data)
                    #im = Image.fromarray(canvas_result)
                    img = cv.cvtColor(
                        canvas_result.image_data, cv.COLOR_RGB2RGBA)
                    with cols[1]:
                        download_button1(img, button, download,
                                         mime_type, key="1.1")
                else:
                    st.image(canvas_result.image_data)
                    print(canvas_result.image_data)
                    #im = Image.fromarray(canvas_result)
                    img = cv.cvtColor(
                        canvas_result.image_data, cv.COLOR_RGB2RGBA)
                    with cols[1]:
                        download_button1(img, button, download,
                                         mime_type, key="1.2")

        if canvas_result.json_data is not None:
            # need to convert obj to str because PyArrow
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=['object']).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)