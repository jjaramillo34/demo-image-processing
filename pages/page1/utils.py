import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
import string
import random
import requests
import shutil
import imutils
import time
import streamlit.components.v1 as components
from datetime import datetime
from streamlit_cropper import st_cropper
from webcolors import hex_to_name
from PIL import Image, ImageColor
from streamlit_drawable_canvas import st_canvas
from duckduckgo_search import ddg_images
from streamlit_embedcode import github_gist, gitlab_snippet

from utils_helpers import (
    convert_rgb_to_names,
    download_button1,
    duckduck_images,
    gist_code,
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
    duckduck_images,
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
    # print(button)
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
                                 mime_type, key="drawing_1.1")

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
            'Drawing pyimagesearch.com Tutorial',
            'https://pyimagesearch.com/2021/01/27/drawing-with-opencv/',)

        source_code(
            'Source Code Gist',
            'https://gist.github.com/jjaramillo34/ad585a865169a0817a3d712d1f091471')

        with st.expander('DuckDuckGo Search Results'):
            st.subheader('More About Drawing')
            scrape_duckduckgo('basic drawing opencv')

        duckduck_images('Sample Drawing Images',
                        'basic drawing opencv')

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

        tutorial_page(
            'Drawing pyimagesearch.com Tutorial',
            'https://pyimagesearch.com/2021/01/27/drawing-with-opencv/',)

        source_code(
            'Source Code Gist',
            'https://gist.github.com/jjaramillo34/ad585a865169a0817a3d712d1f091471')

        with st.expander('DuckDuckGo Search Results'):
            st.subheader('More About Drawing')
            scrape_duckduckgo('drawing opencv logo')

        duckduck_images('Sample Drawing Images',
                        'drawing opencv logo')

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
                    # print(canvas_result.image_data)
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

        tutorial_page(
            'Drawing pyimagesearch.com Tutorial',
            'https://pyimagesearch.com/2021/01/27/drawing-with-opencv/',)

        source_code(
            'Source Code Gist',
            'https://gist.github.com/jjaramillo34/ad585a865169a0817a3d712d1f091471')

        with st.expander('DuckDuckGo Search Results'):
            st.subheader('More About Drawing')
            scrape_duckduckgo('drawable canvas streamlit')

        duckduck_images('Sample Drawing Images',
                        'drawable canvas streamlit')


def cropping():
    cropper_options = st.sidebar.radio(
        "Cropping Options",
        ('Streamlit-Cropper', 'OpenCV Cropper'))
    st.sidebar.markdown('Streamlit is **_really_ cool**.')

    st.set_option('deprecation.showfileUploaderEncoding', False)

    if cropper_options == 'Streamlit-Cropper':
        # Upload an image and set some options for demo purposes
        with st.expander('Streamlit-Cropper', expanded=True):
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

                    # print(type(cropped_img))
                    c = converted(cropped_img)
                    img = cv.cvtColor(c, cv.COLOR_RGB2RGBA)
                    # print(type(img))
                    download_button1(img, button, download,
                                     mime_type, key="cropping_1.1")
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

                # print(type(cropped_img))
                c = converted(cropped_img)
                img = cv.cvtColor(c, cv.COLOR_RGB2RGBA)
                # print(type(img))
                download_button1(img, button, download,
                                 mime_type, key="cropping_1.2")

        tutorial_page(
            'Stramlit-Cropper pyimagesearch.com Tutorial',
            'https://pyimagesearch.com/2021/01/19/crop-image-with-opencv/',)

        source_code(
            'Source Code Gist',
            'https://gist.github.com/jjaramillo34/f9fce0fe5b458918dcfcbcdbaa441e40')

        with st.expander('DuckDuckGo Search Results'):
            st.subheader('More About Cropping')
            scrape_duckduckgo('streamlit-cropper')

        duckduck_images('Sample Streamlit-Cropper Images',
                        'streamlit-cropper')

    else:
        with st.expander('OpenCV Cropper', expanded=True):
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
                c = converted(result)
                download_button1(c, button, download,
                                 mime_type, key="cropping_1.3")

        tutorial_page(
            'Cropping pyimagesearch.com Tutorial',
            'https://pyimagesearch.com/2021/01/19/crop-image-with-opencv/',)

        source_code(
            'Source Code Gist',
            'https://gist.github.com/jjaramillo34/f9fce0fe5b458918dcfcbcdbaa441e40')

        with st.expander('DuckDuckGo Search Results'):
            st.subheader('More About Cropping')
            scrape_duckduckgo('cropping opencv')

        duckduck_images('Sample Cropping Images',
                        'cropping opencv')


def flipping():
    st.header("Flipping Demo")
    img_file = st.file_uploader(
        label='Upload a file', type=['png', 'jpg', 'jpeg'])
    realtime_update = st.sidebar.checkbox(
        label="Update in Real Time", value=True)

    if img_file is not None:
        with st.expander('Flipping OpenCV', expanded=True):
            cols = st.columns(2)
            image = load_image_PIL(img_file)
            image = converted(image)
            with cols[0]:

                st.write(original)
                st.image(image)

            with cols[1]:
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
                    #img = cv.cvtColor(c, cv.COLOR_RGB2RGBA)
                    download_button1(flipped, button, download,
                                     mime_type, key="flipping_1.1")
                    st.markdown("Source Code")
                    github_gist(
                        "https://gist.github.com/jjaramillo34/176fbef9fbae548f3cb683a647798a01", width=800)

                elif st.session_state.count == 2:
                    # flip the image horizontally
                    flipped = cv.flip(image, 1)
                    st.markdown("[INFO] flipping image horizontally...")
                    st.image(flipped)
                    result = Image.fromarray(flipped)
                    download_button1(flipped, button, download,
                                     mime_type, key="flipping_1.2")
                    st.markdown("Source Code")
                    github_gist(
                        'https://gist.github.com/jjaramillo34/cba249f317b17a1c995fd4e121267383', width=800)
                elif st.session_state.count == 3:
                    # flip the image along both axes
                    flipped = cv.flip(image, -1)
                    st.markdown(
                        "[INFO] flipping image horizontally and vertically...")
                    st.image(flipped)
                    result = Image.fromarray(flipped)
                    download_button1(flipped, button, download,
                                     mime_type, key="flipping_1.3")
                    st.markdown("Source Code")
                    github_gist(
                        'https://gist.github.com/jjaramillo34/f073f2cb0263d03b49c08f65897993bd', width=800)
                elif st.session_state.count == 4:
                    st.session_state.count = 0
                    st.markdown("[INFO] Original Image...")
                    st.image(image)

    else:
        with st.expander("Flipping OpenCV Default Image", expanded=True):
            cols = st.columns(2)
            with cols[0]:
                image = load_image(default_image)
                st.write(original)
                st.image(image)

            with cols[1]:
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
                    download_button1(flipped, button, download,
                                     mime_type, key="flipping_1.4")
                    st.markdown("Source Code")
                    github_gist(
                        "https://gist.github.com/jjaramillo34/176fbef9fbae548f3cb683a647798a01", width=800)

                elif st.session_state.count == 2:
                    # flip the image horizontally
                    flipped = cv.flip(image, 1)
                    st.markdown("[INFO] flipping image horizontally...")
                    st.image(flipped)
                    result = Image.fromarray(flipped)
                    download_button1(flipped, button, download,
                                     mime_type, key="flipping_1.1")
                    st.markdown("Source Code")
                    github_gist(
                        'https://gist.github.com/jjaramillo34/cba249f317b17a1c995fd4e121267383', width=800)
                elif st.session_state.count == 3:
                    # flip the image along both axes
                    flipped = cv.flip(image, -1)
                    st.markdown(
                        "[INFO] flipping image horizontally and vertically...")
                    st.image(flipped)
                    st.markdown("Source Code")
                    github_gist(
                        'https://gist.github.com/jjaramillo34/f073f2cb0263d03b49c08f65897993bd', width=800)
                elif st.session_state.count == 4:
                    st.session_state.count = 0
                    st.markdown("[INFO] Original Image...")
                    st.image(image)

    tutorial_page('Flipping pyimagesearch.com Tutorial',
                  'https://pyimagesearch.com/2021/01/20/opencv-flip-image-cv2-flip/')

    source_code(
        'Source Code Gist',
        'https://gist.github.com/jjaramillo34/46bd88b64ac5999be44f0fecd150313b')

    with st.expander('DuckDuckGo Search Results'):
        st.subheader('More About Flipping')
        scrape_duckduckgo('flipping opencv')

    duckduck_images('Sample Flipping Images',
                    'flipping opencv')


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
            with st.expander("Masking Demo Upload", expanded=True):
                cols = st.columns(3)
                with cols[0]:
                    image = load_image_PIL(img_file)
                    image = converted(image)
                    st.markdown(original)
                    st.image(image)
                with cols[1]:
                    # a mask is the same size as our image, but has only two pixels values, 0 to 255 -- pixels with
                    # value of 0 (background) are ignored in the original image while mask pixels with a value 255
                    # (foreground) are allowed to be kept
                    mask = np.zeros(image.shape[:2], dtype="uint8")

                    topLeft = st.slider(
                        'Select a range of TopLeft Corner', 0, image.shape[2], (0, 200))
                    topRight = st.slider(
                        'Select a range of TopRight Corner', 0, image.shape[1], (400, 850))
                    #cv.putText(image, f'({topLeft[0]+ 10}, {topLeft[1]+ 10})', topLeft, font, 4,(255,255,255),2,cv.LINE_AA)
                    cv.rectangle(mask, topLeft, topRight, 255, -1)
                    masked = cv.bitwise_and(image, image, mask=mask)

                    # show the output images
                    #st.markdown('Rectangular Mask')
                    # st.image(mask)
                    st.markdown("Mask applied to Image")
                    st.image(masked)
                    github_gist(
                        'https://gist.github.com/jjaramillo34/209c2c5b7ef203a4b7702717ea55d10a', width=500)

                with cols[2]:
                    # now let's make a circular mask with a radius of 298 pixels and appy the mask again
                    mask = np.zeros(image.shape[:2], dtype="uint8")

                    coor = st.slider('Select a range of Coordinates',
                                     0, image.shape[1], (300, 500))
                    r = st.slider('Select a range of Radius', 0, 800, 298)
                    cv.circle(mask, coor, r, 255, -1)
                    masked = cv.bitwise_and(image, image, mask=mask)

                    # show the output images
                    #st.markdown("Circular Mask")
                    # st.image(mask)
                    st.markdown("Mask Applied to image")
                    st.image(masked)

                    github_gist(
                        'https://gist.github.com/jjaramillo34/7a05e332d9ef300aafe8d48192da2c82', width=500)

            tutorial_page('Masking pyimagesearch.com Tutorial',
                          'https://pyimagesearch.com/2021/01/20/opencv-mask-image-cv2-bitwise_and/')

            source_code(
                'Source Code Gist', 'https://gist.github.com/jjaramillo34/46bd88b64ac5999be44f0fecd150313b')

            with st.expander('DuckDuckGo Search Results'):
                st.subheader('More About Simple Masking')
                scrape_duckduckgo('simple masking opencv')

            duckduck_images('Sample Simple Masking Images',
                            'simple masking opencv')

        else:
            with st.expander('Masking Demo', expanded=True):
                cols = st.columns(3)
                with cols[0]:
                    image = load_image(default_image)
                    st.markdown(original)
                    st.image(image)
                with cols[1]:
                    # a mask is the same size as our image, but has only two pixels values, 0 to 255 -- pixels with
                    # value of 0 (background) are ignored in the original image while mask pixels with a value 255
                    # (foreground) are allowed to be kept
                    mask = np.zeros(image.shape[:2], dtype="uint8")

                    topLeft = st.slider(
                        'Select a range of TopLeft Corner', 0, image.shape[2], (0, 200))

                    topRight = st.slider(
                        'Select a range of TopRight Corner', 0, image.shape[1], (400, 1200))
                    #cv.putText(image, f'({topLeft[0]+ 10}, {topLeft[1]+ 10})', topLeft, font, 4,(255,255,255),2,cv.LINE_AA)
                    cv.rectangle(mask, topLeft, topRight, 255, -1)
                    masked = cv.bitwise_and(image, image, mask=mask)

                    # show the output images
                    #st.markdown('Rectangular Mask')
                    # st.image(mask)
                    st.markdown("Mask applied to Image")
                    st.image(masked)

                    github_gist(
                        'https://gist.github.com/jjaramillo34/209c2c5b7ef203a4b7702717ea55d10a', width=500)

                with cols[2]:
                    # now let's make a circular mask with a radius of 298 pixels and appy the mask again
                    mask = np.zeros(image.shape[:2], dtype="uint8")

                    coor = st.slider('Select a range of Coordinates',
                                     0, image.shape[1], (300, 500))
                    r = st.slider('Select a range of Radius', 0, 800, 298)
                    cv.circle(mask, coor, r, 255, -1)
                    masked = cv.bitwise_and(image, image, mask=mask)
                    # show the output images
                    st.markdown("Mask Applied to image")
                    st.image(masked)
                    github_gist(
                        'https://gist.github.com/jjaramillo34/7a05e332d9ef300aafe8d48192da2c82', width=500)

            tutorial_page('Masking pyimagesearch.com Tutorial',
                          'https://pyimagesearch.com/2021/01/20/opencv-mask-image-cv2-bitwise_and/')

            source_code(
                'Source Code Gist', 'https://gist.github.com/jjaramillo34/46bd88b64ac5999be44f0fecd150313b')

            with st.expander('DuckDuckGo Search Results'):
                st.subheader('More About Simple Masking')
                scrape_duckduckgo('simple masking opencv')

            duckduck_images('Sample Simple Masking Images',
                            'simple masking opencv')

    else:
        if img_file is not None:
            with st.expander('Black/White, Color Masking', expanded=True):
                cols = st.columns(4)
                with cols[0]:
                    color = st.color_picker('Pick A Color', '#FFFF00')
                with cols[1]:
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

                with cols[0]:
                    HMin = st.slider('HMin', min_value=0,
                                     max_value=179, value=20, key='1')
                    SMin = st.slider('SMin', min_value=0,
                                     max_value=255, value=70, key='2')
                    VMin = st.slider('VMin', min_value=0,
                                     max_value=255, value=100, key='3')
                with cols[1]:
                    HMax = st.slider('HMax', min_value=0, max_value=179,
                                     value=int(hsv_color[0][0][0]), key='4')
                    SMax = st.slider('SMax', min_value=0, max_value=255,
                                     value=int(hsv_color[0][0][1]), key='5')
                    VMax = st.slider('VMax', min_value=0, max_value=255,
                                     value=int(hsv_color[0][0][2]), key='6')
                #hMin = sMin = vMin = hMax = sMax = vMax = 0
                #phMin = psMin = pvMin = phMax = psMax = pvMax = 0

                #cols = st.columns(2)

                with cols[2]:
                    # Set minimum and maximum HSV values to display
                    lower = np.array([HMin, SMin, VMin])
                    upper = np.array([HMax, SMax, VMax])

                    image = load_image_PIL(img_file)
                    image = converted(image)
                    st.markdown(original)
                    st.image(image)

                with cols[3]:
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
                    st.markdown("Mask Applied to image")
                    st.image(Mask)

            tutorial_page("Black/White, Color Masking pyimagesearch.com Tutorial",
                          "https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/")

            source_code(
                'Source Code Gist', 'https://gist.github.com/jjaramillo34/7a05e332d9ef300aafe8d48192da2c82')

            with st.expander('DuckDuckGo Search Results', expanded=False):
                st.subheader("More about Black/White, Color Masking")
                scrape_duckduckgo('black, white, color masking opencv')

            duckduck_images('Sample Masking Images',
                            'black, white, color masking opencv')

        else:
            with st.expander('Black/White, Color Masking', expanded=True):
                cols = st.columns(4)
                with cols[0]:
                    color = st.color_picker('Pick A Color', '#FFFF00')
                with cols[1]:
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

                with cols[0]:
                    HMin = st.slider('HMin', min_value=0,
                                     max_value=179, value=20, key='1')
                    SMin = st.slider('SMin', min_value=0,
                                     max_value=255, value=70, key='2')
                    VMin = st.slider('VMin', min_value=0,
                                     max_value=255, value=100, key='3')
                with cols[1]:
                    HMax = st.slider('HMax', min_value=0, max_value=179,
                                     value=int(hsv_color[0][0][0]), key='4')
                    SMax = st.slider('SMax', min_value=0, max_value=255,
                                     value=int(hsv_color[0][0][1]), key='5')
                    VMax = st.slider('VMax', min_value=0, max_value=255,
                                     value=int(hsv_color[0][0][2]), key='6')
                #hMin = sMin = vMin = hMax = sMax = vMax = 0
                #phMin = psMin = pvMin = phMax = psMax = pvMax = 0

                #cols = st.columns(2)

                with cols[2]:
                    # Set minimum and maximum HSV values to display
                    lower = np.array([HMin, SMin, VMin])
                    upper = np.array([HMax, SMax, VMax])

                    image = load_image('images/stop.jpg')
                    st.markdown(original)
                    st.image(image)

                with cols[3]:
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

            tutorial_page("Black/White, Color Masking pyimagesearch.com Tutorial",
                          "https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/")

            source_code(
                'Source Code Gist', 'https://gist.github.com/jjaramillo34/7a05e332d9ef300aafe8d48192da2c82')

            with st.expander('DuckDuckGo Search Results', expanded=False):
                st.subheader("More about Black/White, Color Masking")
                scrape_duckduckgo('black, white, color masking opencv')

            duckduck_images('Black/White, Color Masking Images',
                            'black, white, color masking opencv')
