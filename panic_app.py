import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import base64
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import MinMaxScaler
d = {'female': {0.0, 1.0},
 'smoker': {0.0, 1.0},
 'alcohol': {0.0, 1.0},
 'nutrition': {0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 9.0},
 'prior': {0.0, 1.0},
 'leukocytosis': {0.0, 1.0},
 'steroids': {0.0, 1.0},
 'asa': {1.0, 2.0, 3.0, 4.0, 5.0},
 'renal': {0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
 'hand': {0.0, 1.0},
 'emergent': {0.0, 1.0},
 'laparoscopic': {1.0, 2.0, 3.0},
 'type': {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
 'indication': {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
 'perforation ': {0.0, 1.0},
 'ct': {0.0, 1.0, 2.0, 3.0, 4.0},
 'cn': {0.0, 1.0, 2.0},
 'cm': {0.0, 1.0},
 'livermets': {0.0, 1.0},
 'ai': {0.0, 1.0}}
page = st.sidebar.selectbox("Select Activity", ["Panic Prediction",])
st.sidebar.text(" \n")


pkl_file1 = open('min_max_scaler1.pkl', 'rb')
scaler = pickle.load(pkl_file1)


pkl_file2 = open('rfc.pkl', 'rb')
rfc = pickle.load(pkl_file2)


if page=="Panic Prediction":

    st.header("Panic Prediction")
    st.text(" \n")

    form = st.form(key='my_form2')

    x1 = form.text_input(label='cci')
    form.text(" \n")

    x2 = form.text_input(label='age')
    form.text(" \n")


    x3 = form.text_input(label='albumin')
    form.text(" \n")


    x4 = form.text_input(label='bmi')
    form.text(" \n")

    x5 = form.text_input(label='hemoglobin')
    form.text(" \n")



    x6 = form.selectbox('alcohol',["YES","NO"], key=1)
    form.text(" \n")

    x7 = form.selectbox('asa',list(d["asa"]),key=1)
    form.text(" \n")


    x8 = form.selectbox("Gender",["MALE","FEMALE"],key=1)
    form.text(" \n")


    x9 = form.selectbox('leukocytosis',["YES","NO"],key=1)
    form.text(" \n")


    x10 = form.selectbox('nutrition',list(d['nutrition']),key=1)
    form.text(" \n")

    x11 = form.selectbox('prior',["YES","NO"],key=1)
    form.text(" \n")

    x12 = form.selectbox("renal",list(d["renal"]),key=1)
    form.text(" \n")




    x13 = form.selectbox("smoker",["YES","NO"],key=1)
    form.text(" \n")



    x14 = form.selectbox("steroids",["YES","NO"],key=1)
    form.text(" \n")


    submit_button = form.form_submit_button(label='Predict Panic')



    if submit_button:
        l = {"YES":1,"NO":0}
        f = {"MALE":0,"FEMALE":1}
        x1 =  float(x1)
        x2 = float(x2)
        x3 = float(x3)
        x4 = float(x4)
        x5 = float(x5)

        x6 = int(l[x6])
        x7 = int(x7)
        x8 = int(f[x8])
        x9 = int(l[x9])
        x10 = int(x10)
        x11 = int(l[x11])
        x12 = int(x12)
        x13 = int(l[x13])

        x14 = int(l[x14])

        o = scaler.transform([[x1,x2,x3,x4,x5]])
        x1,x2,x3,x4,x5 = o[0][0],o[0][1],o[0][2],o[0][3],o[0][4]


        c1 = rfc.predict([[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14]])

        c2 = rfc.predict_proba([[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14]])



        st.header("probability to get panic :")
        st.text(c2[1])
        st.header("Predict output")
        st.text(c1[0])
