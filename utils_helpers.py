
from time import sleep
import cv2
import streamlit as st
import pandas as pd
import sqlite3
from sqlite3 import Error, adapters
import numpy as np
import base64
import sys
import os
import re
import uuid
import time
import ipinfo
import pymongo
from datetime import datetime
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
from skimage.exposure import rescale_intensity
from duckduckgo_search import ddg
from io import BytesIO
from PIL import Image
from scipy.spatial import KDTree
import streamlit.components.v1 as components
from streamlit_embedcode import github_gist, gitlab_snippet

def version():
    return st.sidebar.caption(f"Streamlit version `{st.__version__}`")

def load_image(filename):
    image = cv2.imread(filename)
    return image

def load_image_PIL(filename):
    image = Image.open(filename)
    return image

def converted(image):
    #image = cv2.imread(filename)
    converted_image = np.array(image.convert('RGB'))
    return converted_image

def increment_counter():
	st.session_state.count += 1
 
def load_image_file_uploader(image):
    img_file = st.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])
    image = load_image_PIL(img_file)
    image = converted(image)
    return st.image(image)

def source_code(title, url, gist_url):
    with st.expander(title):
        cols = st.columns(2)
        with cols[0]:
            components.iframe(url, height=800, scrolling=True)
        with cols[1]:
            github_gist(gist_url, width=800, height=800)
    st.sidebar.caption(f"Streamlit version `{st.__version__}`")
    
def tutorial(title, url):
    with st.expander(title):
        components.iframe(url, height=800, scrolling=True)

def gist_code(title, gist_url):
    with st.expander(title):
        github_gist(gist_url, height=800)
    st.sidebar.caption(f"Streamlit version `{st.__version__}`")

def gitlab_code(title, gitlab_url):
    with st.expander(title):
        gitlab_snippet(gitlab_url, height=800)
    st.sidebar.caption(f"Streamlit version `{st.__version__}`")

def convert_rgb_to_names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'

def auto_canny_thresh(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

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
	return output

def download_button1(image, label, file_name, mime, key):
    # comvert np.array into PIL image
    result = Image.fromarray(image)
        
    buf = BytesIO()
    result.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    btn = st.download_button(
        label=label,
        data=byte_im,
        file_name=file_name,
        mime=mime, key=key)
    return btn

def download_button(object_to_download, download_filename, button_text, isPNG):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(Pillow_image_from_cv_matrix, 'your_image.jpg', 'Click to me to download!')
    """

    buffered = BytesIO()
    if isPNG:
        object_to_download.save(buffered, format="PNG")
    else:
        object_to_download.save(buffered, format="JPEG")
    b64 = base64.b64encode(buffered.getvalue()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 

            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    return dl_link

# returns dictinary with ip location
def get_location_data():
    import urllib.request
    external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
    #print(external_ip)
    ACCESS_TOKEN = st.secrets["secret"]["secret"]
    #print('Token', ACCESS_TOKEN)
    handler = ipinfo.getHandler(ACCESS_TOKEN)
    details = handler.getDetails(external_ip)
    print(details.details)
    return details.details

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    return pymongo.MongoClient(**st.secrets["mongo_ratings"])

client = init_connection()    

@st.experimental_memo(ttl=600)
def insert_data_mongodb(rating, feedback, date_r, city, ip, region, country, loc):
    
    #client = pymongo.MongoClient(st.secrets["mongo_ratings"]['host'])
        
    print("MongoDB Connected successfully!!!")
    # database
    database = client['app_ratings']
    # Created collection
    collection = database['ratings']
    
    location_dict = get_location_data()
    
    date_r = datetime.now()
    loc = location_dict['loc']
    city = location_dict['city']
    ip = location_dict['ip']
    region = location_dict['region']
    country = location_dict['country']
    my_dict = {
        "rating": rating, 
        "feedback": feedback, 
        'date': date_r,
        'city': city,
        'ip': ip,
        'region': region,
        'country': country,
        'loc' : { 'type': "Point", 'coordinates': [loc.split(',')[0] ,loc.split(',')[1]] },
        }
    x = collection.insert_one(my_dict)
    #client.close()
    print("MongoDB Close successfully!!!")
    
@st.experimental_singleton
def average_ratings_mongodb():
    
    print("MongoDB Connected successfully!!!")
    # database
    #client = pymongo.MongoClient(st.secrets["mongo_ratings"]['host'])
    database = client['app_ratings']
    # Created collection
    collection = database['ratings']
    
    x = collection.aggregate([
        {"$group": 
            {
                "_id": None, 
                "avg_rating": {"$avg":"$rating"} } 
            }
    ])
    #client.close()
    print("MongoDB Close successfully!!!")
    return list(x)[0]['avg_rating']

#@st.experimental_singleton(suppress_st_warning=True)     
def scrape_duckduckgo(col_name):
    # Python code to illustrate inserting data in MongoDB

    print("Connected successfully!!!")
    # database
    #client = pymongo.MongoClient(st.secrets["mongo_ratings"]['host'])
    db = client.duckduckgo
    # Created collection
    c = col_name.replace(" ", '_')
    collection = db[c]
    if c in db.list_collection_names():
        print("The collection exists.")
        # This is a cursor instance
        cur = collection.find()   
        results = list(cur)
        
        # Checking the cursor is empty
        # or not
        if len(results)==0:
            print("Empty Cursor")
            keywords = col_name
            results = ddg(keywords, region='wt-wt', safesearch='Moderate', time='y', max_results=10)
            time.sleep(0.75)
            result_df = pd.DataFrame.from_dict(results)
            result_df.reset_index(inplace=True)
            data_dict = result_df.to_dict("records")
            collection.insert_many(data_dict)
            st.dataframe(result_df, height=850)
        else:
            print("Cursor is Not Empty")
            print("Do Stuff Here")
            cols = st.columns(2)
            results = collection.find({})
            with cols[0]:
                for doc in results:
                    #st.write(doc['title'])
                    st.markdown(f'<h5 style="font-size:12px;"><a href="{doc["href"]}" target="_blank">{doc["title"]}</a></h5>', unsafe_allow_html=True)
            with cols[1]:
                results_df = pd.DataFrame(list(collection.find({}, {'_id': False})))
                st.dataframe(results_df, height=850)
                csv = convert_df(results_df)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='results.csv',
                    mime='text/csv')
    else:
        collection = db.create_collection(c)
        print("The collection not exists collection has been creaed.")
        keywords = col_name
        results = ddg(keywords, region='wt-wt', safesearch='Moderate', time='y', max_results=10)
        time.sleep(0.75)
        result_df = pd.DataFrame.from_dict(results)
        result_df.reset_index(inplace=True)
        data_dict = result_df.to_dict("records")
        collection.insert_many(data_dict)
        st.dataframe(result_df, height=850)
        
    if col_name in db.list_collection_names():
        print(True)
    #else:
    #    with st.spinner('Getting results from duckduckgo...'):
    #        keywords = col_name
    #        results = ddg(keywords, region='wt-wt', safesearch='Moderate', time='y', max_results=10)
    #        time.sleep(0.75)
    #        #st.write(results)
    #        cols = st.columns(2)
    #        with cols[0]:
    #            for i, ele in enumerate(results):
    #                st.markdown(f'<h5 style="font-size:12px;"><a href="{results[i]["href"]}" target="_blank">{results[i]["title"]}</a></h5>', unsafe_allow_html=True)
    #        with cols[1]:
    #            result_df = pd.DataFrame.from_dict(results)
    #            st.dataframe(result_df, height=850)
    #            csv = convert_df(result_df)
    #            st.download_button(
    #                label="Download data as CSV",
    #                data=csv,
    #                file_name='results.csv',
    #                mime='text/csv',)
            
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


