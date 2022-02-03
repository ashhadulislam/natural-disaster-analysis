import streamlit as st
import numpy as np
import pandas as pd

import random

import plotly.express as px
import os
from collections import Counter
import plotly.graph_objects as go


# @st.cache
def app():
    header=st.container()
    gender_distn = st.container()
    age_distn = st.container()
    top_subjects = st.container()
    important_subjects = st.container()
    footer = st.container() 


    with header:
        # st.title("Private tutors in Qatar")
        st.subheader("Analyse and understand the skillsets needed to succeed")    

    with gender_distn:
        st.header("Number of male and female tutors in Qatar")
        # st.subheader("Please choose the dates")
        df_tutors=pd.read_csv("data/mpt_data_tutors_details.csv")
        df_gender_group=df_tutors.groupby(["Gender"]).count().reset_index()
        df_gender_group=df_gender_group[df_gender_group["Gender"]!="None"]
        fig = px.bar(df_gender_group, x='Gender',y="Name",color='Gender')


        fig.update_layout(
            title="Distribution of tutors based on Gender",
            xaxis_title="Gender",
            yaxis_title="Count",
        #     legend_title="Legend Title",
        #     font=dict(
        #         family="Courier New, monospace",
        #         size=18,
        #         color="RebeccaPurple"
        #     )
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with age_distn:
        st.header("Distribution of age")
        st.subheader("Tutors in Doha according to age group")
        ages=list(df_tutors["Age"])
        good_ages=[]
        for age in ages:
            if age.isdigit():
                if int(age)>0 and int(age)<100:
                    good_ages.append(int(age))
                
        good_ages=np.array(good_ages)        

        counts, bins = np.histogram(good_ages)
        bins = 0.5 * (bins[:-1] + bins[1:])

        fig = px.bar(x=bins, y=counts, labels={'x':'age', 'y':'count'})
        
        st.plotly_chart(fig, use_container_width=True)

    with top_subjects:
        st.header("Subjects having maxximum number of trainers")
        st.subheader("Top ranking subjects common among tutors")
        print(df_tutors.shape,"is the size of the file")
        subjects_taught=list(df_tutors["SubjectsTaught"])
        all_subjects_taught=[]
        for subjects in subjects_taught:
            subjects_list=subjects.split(";")
            all_subjects_taught.extend(subjects_list)

        all_subjects_taught = np.array(all_subjects_taught)
        all_subjects_taught = np.where(all_subjects_taught == 'MATHS', 'Mathematics', all_subjects_taught)
        all_subjects_taught=list(all_subjects_taught)                
        counter_subjects=Counter(all_subjects_taught)
        sorted_counter_subjects={k: v for k, v in sorted(counter_subjects.items(), key=lambda item: item[1],reverse=True)}

        print("sorted subjects are ",sorted_counter_subjects)


        top_count=st.selectbox("Number of top subjects", [i for i in range(1,len(list(sorted_counter_subjects.keys())))],index=5)

        colors=[]
        col1=random.uniform(0,1)
        col2=random.uniform(col1,col1+0.05)
        for i in range(top_count):
            newcolor=random.uniform(col1,col2)    
            colors.append(newcolor)
            col1=newcolor
            col2=newcolor+0.05
            
            
        top_subjects=list(sorted_counter_subjects.keys())[:top_count]
        top_subject_counts=list(sorted_counter_subjects.values())[:top_count]
        print("Colors are ",colors)    

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_subjects,
            y=top_subject_counts,
            marker_color=colors
        ))

        fig.update_layout(
            title="Top Subjects by trainer count",
            xaxis_title="Subject",
            yaxis_title="Count Of Trainers",
        #     legend_title="Legend Title",
        #     font=dict(
        #         family="Courier New, monospace",
        #         size=18,
        #         color="RebeccaPurple"
        #     )
        )
        st.plotly_chart(fig, use_container_width=True)


    with important_subjects:
        st.header("Subjects with high demand")
        st.subheader("Yet less number of tutors.")

        df_advert_subjwise=pd.read_csv("data/advert_by_subject.csv")
        advert_subjets=list(df_advert_subjwise.Subjects)
        count_advert_subjects=list(df_advert_subjwise["Count Of Adverts"])
        dic_advert_subjects={}
        for i in range(len(advert_subjets)):
            dic_advert_subjects[advert_subjets[i]]=count_advert_subjects[i]            


        subjects_to_be_dropped=[]
        for advert_subj in dic_advert_subjects.keys():
            if advert_subj not in list(sorted_counter_subjects.keys()):
        #         print(advert_subj,"not in tutor list and its value is ",dic_advert_subjects[advert_subj])   
                subjects_to_be_dropped.append(advert_subj)

        for subj_to_drop in subjects_to_be_dropped:
            del dic_advert_subjects[subj_to_drop]
        weights_advertised_subjects=np.linspace(1,0,len(dic_advert_subjects))
        weights_subjects_tutors=np.linspace(0,1,len(sorted_counter_subjects))


        # for each advertised subject, find its position in the advert board and in the
        # tutor board
        # multiply weight of the positions
        # assign that as the score

        subject_score={}
        advert_index=0
        for advert_subj in list(dic_advert_subjects.keys()):
            wt_this_advt_subj=weights_advertised_subjects[advert_index]
            tutor_index=0
            for tutor_subj in list(sorted_counter_subjects.keys()):
                if advert_subj==tutor_subj:
                    wt_this_tutor_subject=weights_subjects_tutors[tutor_index]
                    total_wt=wt_this_advt_subj*wt_this_tutor_subject
                    subject_score[advert_subj]=total_wt
                    break
                
                tutor_index+=1
            
            advert_index+=1
        sorted_subject_score={k: v for k, v in sorted(subject_score.items(), key=lambda item: item[1],reverse=True)}            
        top_count=st.selectbox("Number of most important subjects", [i for i in range(1,len(list(sorted_subject_score.keys())))],index=5)    




        colors=[]
        col1=random.uniform(0,1)
        col2=random.uniform(col1,col1+0.05)
        for i in range(top_count):
            newcolor=random.uniform(col1,col2)    
            colors.append(newcolor)
            col1=newcolor
            col2=newcolor+0.05
            
        
        top_subjects=list(sorted_subject_score.keys())[:top_count]
        top_subject_counts=list(sorted_subject_score.values())[:top_count]
        print("Colors are ",colors)    

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_subjects,
            y=top_subject_counts,
            marker_color=colors
        ))

        fig.update_layout(
            title="Top Subjects by importance (High demand, low supply)",
            xaxis_title="Subject",
            yaxis_title="Importance Score",
        #     legend_title="Legend Title",
        #     font=dict(
        #         family="Courier New, monospace",
        #         size=18,
        #         color="RebeccaPurple"
        #     )
        )
        st.plotly_chart(fig, use_container_width=True)



    with footer:
        st.header("Read the detailed discussion on ")
        st.write("[Medium](https://ashhadulislam.medium.com/freelance-tutoring-in-qatar-33a27bee1403)")

