import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image,ImageEnhance
from datetime import datetime
from io import BytesIO
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

uploaded_files = st.file_uploader("Choose a jpg file",type=['png', 'jpg','jpeg'], accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
images_list=uploaded_files

# # @st.cache
model=tf.keras.models.load_model('model',compile = True)
# # Output label of good or bad for the model.
label=None

# Initialize the session state to have the current image index=0
if 'img_idx' not in st.session_state:
    st.session_state.img_idx=0

# Ensure img_idx will always be within images_list
if st.session_state.img_idx > (len(images_list)):
    st.session_state.img_idx = (len(images_list)-1) if (len(images_list)-1)>0 else 0


def next_button():
    if -1 < st.session_state.img_idx <= (len(images_list)-1)   :
        st.session_state.img_idx += 1
    elif st.session_state.img_idx ==(len(images_list)):
        st.success('All images have been sorted!')
    else:
        st.warning(f'No more images to sort { st.session_state.img_idx} /{ len(images_list)} ')


def back_button():
    if st.session_state.img_idx >0:
        st.session_state.img_idx -= 1
    else:
        st.warning('Cannot Undo')

if images_list==[]:
    image= Image.open("./assets/new_loading_sniffer.jpg")
else:
    if st.session_state.img_idx>=len(images_list):
        image = Image.open("./assets/done.jpg")
    else:
        image = Image.open(images_list[st.session_state.img_idx])


# Sets num_image=1 if images_list is empty
num_images=(len(images_list)) if (len(images_list))>0 else 1


try:
    my_bar = st.progress((st.session_state.img_idx)/num_images)
except st.StreamlitAPIException:
    my_bar = st.progress(0)


col1,col2,col3,col4=st.columns(4)
with col1:
    st.button(label="Next",key="next_button",on_click=next_button)
    st.button(label="Back",key="back_button",on_click=back_button)
    
with col2:
    # Display done.jpg when all images are sorted 
    if st.session_state.img_idx>=len(images_list):
        image = Image.open("./assets/done.jpg")
        st.image(image,width=300)
    else:
        # caption is "" when images_list is empty otherwise its image name 
        caption = '' if images_list==[] else f'#{st.session_state.img_idx} {images_list[st.session_state.img_idx].name}'
        st.image(image, caption=caption,width=300)


def pre_process_img(image):
    size=(250,500)
    img=image.resize(size,Image.ANTIALIAS)
    imgArray = np.asarray(img)
    imgArray=imgArray.reshape((1,250,500,3))# Create batch axis
    return imgArray


def run_predict():
    img_index=st.session_state.img_idx
    # download the last image if the user has already seen it
    if img_index>=len(images_list):
        img_index=(len(images_list)-1)
    # Make sure the images list is not empty and the index is valid
    if 0<=img_index<(len(images_list)) and images_list !=  []:
        img=images_list[img_index]
        img = Image.open(img)
        img_array=pre_process_img(img)
        prediction=model.predict(img_array)
        reverse_mapping={0:"good",1:"bad"}
        # # Transform the array of predictions into a 1d array then return the index of the column with the max prediction
        # # Use the max prediction as the key to the label dictionary
        label=reverse_mapping[prediction.flatten().argmax(axis=0)]
        st.write(label)


with col4:
    st.button(label="Predict Image",key="predict_button",on_click=run_predict)
    if label:
        label

