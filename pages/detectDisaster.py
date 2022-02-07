import streamlit as st
import numpy as np
import pandas as pd

import random

import plotly.express as px
import os
from collections import Counter
import plotly.graph_objects as go
from lib import commons
import torch

# @st.cache
def app():
    header=st.container()
    detect_disaster_type = st.container()    
    footer = st.container() 


    with header:
        # st.title("Private tutors in Qatar")
        st.subheader("Find out the type of disaster affecting a region")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                          "filesize":image_file.size}
            # st.write(file_details)

            # To View Uploaded Image
            st.image(commons.load_image(image_file)
                ,width=250
                )
            print("Image file is it showing location?",image_file)
            image_for_model = commons.image_loader(image_file)
            print("Loaded image for model")
        else:
            proxy_img_file="data/joplin-tornado_00000001_post_disaster.png"
            st.image(commons.load_image(proxy_img_file),width=250)
            image_for_model=commons.image_loader(proxy_img_file)
            print("Loaded proxy image for model")

    with detect_disaster_type:
        model_fulls=os.listdir("models/detect_disaster_type")
        dic_models={}
        for model_full in model_fulls:
            just_name=model_full.split("_")[0]
            dic_models[just_name]=model_full
        print(dic_models)
        model_name=st.selectbox('Please choose the model', options=dic_models.keys(), index = 0)
        model_file_name=dic_models[model_name]
        num_classes = 7        
        batch_size = 32
        num_epochs = 10
        feature_extract = False
        # Initialize the model for this run
        model_ft, input_size = commons.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)        
        model_state_path="models/detect_disaster_type/"+model_file_name
        if torch.cuda.is_available():
            model_ft.load_state_dict(torch.load(model_state_path))
        else:
            model_ft.load_state_dict(torch.load(model_state_path,map_location=torch.device('cpu')))
        res=model_ft(image_for_model)
        _, pred = torch.max(res, 1)
        reverse_map_dict={
                            0: 'fire',
                            1: 'volcano',
                            2: 'flooding',
                            3: 'bushfire',
                            4: 'tsunami',
                            5: 'wildfire',
                            6: 'tornado'
                        }
        disaster_type=reverse_map_dict[pred[0].item()]
        st.subheader("The area was affected by "+disaster_type)
        

    with footer:
        st.header("Read the detailed discussion on ")
        # st.write("[Medium](https://ashhadulislam.medium.com/freelance-tutoring-in-qatar-33a27bee1403)")

