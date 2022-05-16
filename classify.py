import streamlit as st
# import numpy as np
from numpy import asarray,round
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy  as np
from skimage.transform import resize
from streamlit_option_menu import option_menu
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Load the tensorflow model only once
@st.experimental_singleton
def get_model(model_name:str,model_path:str=os.getcwd()+ os.sep+"models"):
    model_path+=os.sep+model_name
    model = tf.keras.models.load_model(model_path+os.sep+"model")
    return model


# Load the model from the singleton cache
model=get_model("binary_classification_model_v_2_1")
# Initialize the  states
if 'label' not in st.session_state:
    st.session_state.label=None
if 'prediction' not in st.session_state:
    st.session_state.prediction=None
# Initialize the session state to have the current image index=0
if 'img_idx' not in st.session_state:
    st.session_state.img_idx=0
if 'prediction_df' not in st.session_state:
    st.session_state.prediction_df=pd.DataFrame(columns=["Filename","Predicted_Label","Probability"])


uploaded_files = st.file_uploader("Choose a jpg file",type=['png', 'jpg','jpeg'], accept_multiple_files=True,key="upload_files_comp")
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
images_list=uploaded_files

with st.sidebar:
    choose = option_menu("App Gallery", ["Classifier", "View Predictions", "Label Reviewer"],
                         icons=['images', 'basket2', 'kanban'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "blue", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )


if choose == "Classifier":

    # Ensure img_idx will always be within images_list
    if st.session_state.img_idx > (len(images_list)+2):
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

    # Display a the image at the current img_idx
    with col2:
        # Display a the done image when the all images have been displayed
        if st.session_state.img_idx>=len(images_list):
            image = Image.open("./assets/done.jpg")
            st.image(image,width=300)
        else:
            st.write("Index",st.session_state.img_idx)
            if st.session_state.prediction_df.empty ==False:
                curr_pred=st.session_state.prediction_df[st.session_state.prediction_df["Filename"]==images_list[st.session_state.img_idx].name]
                st.write("Predicted as ",curr_pred["Predicted_Label"].iloc[0],"with ",curr_pred["Probability"].iloc[0])
            # caption is "" when images_list is empty otherwise its image name 
            caption = '' if images_list==[] else f'#{st.session_state.img_idx} {images_list[st.session_state.img_idx].name}'
            st.image(image, caption=caption,width=300)


    def pre_process_img(image,img_shape:tuple)->list:
        """returns np.array resized to (1,img_shape,3)"""
        img=image.resize(img_shape,Image.ANTIALIAS)
        imgArray = asarray(img)
        imgArray=imgArray.reshape((1,)+img_shape+(3,))# Create batch axis
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
            img_shape=(100,100)
            img_array=pre_process_img(img,img_shape)
            predictions=model.predict(img_array)
            reverse_mapping={0:"bad",1:"good"}
            # # Transform the array of predictions into a 1d array then return the index of the column with the max prediction
            # # Use the max prediction as the key to the label dictionary
            st.session_state.label=reverse_mapping[predictions[0][0].astype('uint8')]
            st.session_state.prediction=round(predictions[0][0],decimals=5)


    def create_prediction_ready_data(images_list:list,img_shape:tuple):
        """returns a numpy array of the shape (number_images,img_shape,3)"""
        images=[]
        for image in images_list:
            img=Image.open(image)
            img_array=pre_process_img(img,img_shape)
            images.append(img_array)
        data=np.vstack(images)
        return data


    def create_predictions_df(predictions):
        labeled_predictions=list(map(lambda x: "good "if x>=0.5 else "bad",predictions))
        for i,img in enumerate(images_list):
            row={"Filename":img.name,'Predicted_Label':labeled_predictions[i],'Probability':predictions[i]}
            st.session_state.prediction_df=pd.concat([st.session_state.prediction_df,pd.DataFrame.from_records([row])],ignore_index=True)


    def run_predict_all():
        # Make sure the images list is not empty and the index is valid
        if images_list !=  []:
            img_shape=(100,100)
            data=create_prediction_ready_data(images_list,img_shape)
            # img_array=pre_process_img(img,img_shape)
            predictions=model.predict(data)
            predictions=predictions.flatten().tolist()
            create_predictions_df(predictions)


    with col4:
        st.button(label="Predict Image",key="predict_button",on_click=run_predict)
        if st.session_state.prediction!=  None:
            st.write(f"Prediction: {st.session_state.prediction}")
        if st.session_state.label != None:
            st.write("Label:",st.session_state.label)

        st.button(label="Predict All Images",key="predict_all_button",on_click=run_predict_all)

    with st.expander("View Predictions"):
        if  st.session_state.prediction_df.empty ==False:       
            st.write(st.session_state.prediction_df)

elif choose == "View Predictions":
    if st.session_state.prediction_df.empty ==False:
        images_per_row=st.slider("The number of images per row",step=1,value=4,min_value=1,max_value=8)
        n_rows=len(images_list)/images_per_row
        n_rows=int(np.ceil(n_rows)) #round up because range is end exclusive
        for row_num in range(n_rows):
            cols=st.columns(images_per_row)
            start=row_num*images_per_row
            end=start+images_per_row
            if end>len(images_list):
                end = len(images_list)
            for col,image in zip(cols,images_list[start:end]):
                curr_pred=st.session_state.prediction_df[st.session_state.prediction_df["Filename"]==image.name]
                predicted_label=curr_pred["Predicted_Label"].iloc[0]
                col.image(image,use_column_width=True,caption=f"{predicted_label}")

