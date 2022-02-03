import streamlit as st
import numpy as np
import pandas as pd
import random

import plotly.express as px
import os
from collections import Counter
import plotly.graph_objects as go
from PIL import Image

from lib import commons
import torch

def load_image(image_file):
    img = Image.open(image_file)
    return img

# @st.cache
def app():
    header=st.container()
    result_all = st.container()
    result_disaster_type = st.container()
    
    footer = st.container() 


    with header:
        # st.title("Private tuition requirements in Qatar over time")
        st.subheader("Test whether an area is affected by any natural disaster")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                          "filesize":image_file.size}
            # st.write(file_details)

            # To View Uploaded Image
            st.image(load_image(image_file)
                # ,width=250
                )
            print("Image file is it showing location?",image_file)
            image_for_model = commons.image_loader(image_file)
            print("Loaded image for model")
        else:
            proxy_img_file="data/joplin-tornado_00000001_post_disaster.png"
            st.image(load_image(proxy_img_file),width=250)
            image_for_model=commons.image_loader(proxy_img_file)
            print("Loaded proxy image for model")

        

    with result_all:        
        model_fulls=os.listdir("models/all")
        dic_models={}
        for model_full in model_fulls:
            just_name=model_full.split("_")[0]
            dic_models[just_name]=model_full
        print(dic_models)
        model_name=st.selectbox('Please choose the model', options=dic_models.keys(), index = 0)


        model_file_name=dic_models[model_name]
        num_classes = 2        
        batch_size = 32
        num_epochs = 10
        feature_extract = False
        # Initialize the model for this run
        model_ft, input_size = commons.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)        
        model_state_path="models/all/"+model_file_name
        
        if torch.cuda.is_available():
            model_ft.load_state_dict(torch.load(model_state_path))
        else:
            model_ft.load_state_dict(torch.load(model_state_path,map_location=torch.device('cpu')))
        res=model_ft(image_for_model)
        _, pred = torch.max(res, 1)
        if pred == 0:
            result="No, this area has not been hit by a disaster"
        elif pred == 1:
            result = "Yes, this area has been hit by a disaster"

        st.subheader(result)
        st.text("To get better results, choose the disaster type below")




        


    with result_disaster_type:   
        st.header("Analyze Satellite Images by disaster type") 
        col1, col2 = st.columns(2) 
        dis_types_list=os.listdir("models/disaster_types")       
        disaster_tp_name = col1.selectbox('Select Disaster Type', options=dis_types_list, index = 0)
        
        dis_based_model_dic={}
        model_dis_based_fulls=os.listdir("models/disaster_types/"+disaster_tp_name)
        print("MOdels are ",model_dis_based_fulls)
        for model_full in model_dis_based_fulls:
            just_name=model_full.split("_")[0]
            dis_based_model_dic[just_name]=model_full
        print(dis_based_model_dic)

        model_name_dis_type = col2.selectbox('Select model', options=dis_based_model_dic.keys(), index = 0)

        
        
        



        image_file2 = st.file_uploader("Upload Satellite Images", type=["png","jpg","jpeg"])
        if image_file2 is not None:
            # To See details
            file_details = {"filename":image_file2.name, "filetype":image_file2.type,
                          "filesize":image_file2.size}
            # st.write(file_details)

            # To View Uploaded Image
            st.image(load_image(image_file2),width=250)
            print("Image file is it showing location?",image_file2)
            image_for_model = commons.image_loader(image_file2)
            print("Loaded image for model")
        else:
            # take image from above section
            if image_file is not None:                
                # To See details
                file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
                # st.write(file_details)

                # To View Uploaded Image
                st.image(load_image(image_file),width=250)
                print("Image file is it showing location?",image_file)
                image_for_model = commons.image_loader(image_file)
                print("Loaded image for model")                
            else:
                proxy_img_file="data/joplin-tornado_00000001_post_disaster.png"
                st.image(load_image(proxy_img_file),width=250)
                image_for_model=commons.image_loader(proxy_img_file)
                print("Loaded proxy image for model")



        model_file_name=dis_based_model_dic[model_name_dis_type]
        num_classes = 2        
        batch_size = 32
        num_epochs = 10
        feature_extract = False
        # # Initialize the model for this run
        model_ft, input_size = commons.initialize_model(model_name_dis_type, num_classes, feature_extract, use_pretrained=True)        
        model_state_path="models/disaster_types/"+disaster_tp_name+"/"+model_file_name
        if torch.cuda.is_available():
            model_ft.load_state_dict(torch.load(model_state_path))
        else:
            model_ft.load_state_dict(torch.load(model_state_path,map_location=torch.device('cpu')))
        res=model_ft(image_for_model)
        _, pred = torch.max(res, 1)
        if pred == 0:
            result="No, this area has not been hit by a disaster"
        elif pred == 1:
            result = "Yes, this area has been hit by a disaster"

        st.subheader(result)


      

        
        


    with footer:
        st.subheader("Project submitted to devpost ")
        st.subheader("Read the detailed discussion ")
        # st.write("[Medium](https://ashhadulislam.medium.com/freelance-tutoring-in-qatar-33a27bee1403)")

