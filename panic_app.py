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

pkl_file2 = open('rfc.pkl', 'rb')
rfc = pickle.load(pkl_file2)


if page=="Panic Prediction":

    st.header("Panic Prediction")
    st.text(" \n")

    form = st.form(key='my_form2')

    x1 = form.slider(label='Charlson Comorbidity Index (CCI)', min_value = 0, max_value = 20, step = 1, value = 4)
    form.text(" \n")

    x2 = form.text_input(label='Age', min_value = 18, max_value = 105, step = 1, value = 65)
    form.text(" \n")


    x3 = form.text_input(label='Albumin', min_value = 0.0, max_value = 10.0, step = 0.1, value = 3.0)
    form.text(" \n")


    x4 = form.text_input(label='Body Mass Index (BMI)', min_value = 15.0, max_value = 60.0, step = 0.1, value = 26.0)
    form.text(" \n")

    x5 = form.text_input(label='Hemoglobin level (in g/dL)', min_value = 0.0, max_value = 150.0, step = 0.1, value = 13.0)
    form.text(" \n")



    x6 = form.selectbox(' Alcohol abuse (>2 alcoholic beverages per day)',["YES","NO"], key=1)
    form.text(" \n")

    x7 = form.selectbox('American Society of Anesthesiologists (ASA) Score',list(d["asa"]),key=1)
    form.text(" \n")


    x8 = form.selectbox("Gender",["MALE","FEMALE"],key=1)
    form.text(" \n")


    x9 = form.selectbox('Leukocytosis',["YES","NO"],key=1)
    form.text(" \n")


    x10 = form.selectbox('Nutritional status (NRS ≥ 3)',list(d['nutrition']),key=1)
    form.text(" \n")

    x11 = form.selectbox('Prior',["YES","NO"],key=1)
    form.text(" \n")

    x12 = form.selectbox("Renal function (CKD Stages G1 (normal) to G5)",list(d["renal"]),key=1)
    form.text(" \n")




    x13 = form.selectbox(" Active smoking",["YES","NO"],key=1)
    form.text(" \n")



    x14 = form.selectbox("Preoperative steroid use (mg)",["YES","NO"],key=1)
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

        df = pd.DataFrame([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14])
        df = (df-df.min())/(df.max()-df.min())
        
        c1 = rfc.predict_proba(df)[:, 1] > 0.10571997452882279

        c2 = rfc.predict_proba(df)[:, 1]



        st.header("probability to get panic :")
        st.text(c2)
        st.header("Predict output")
        st.text(c1)