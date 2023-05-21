import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB

st.write(
"""
    # Web App Klasifikasi Menggunakan Metode KNN
"""
)

st.sidebar.header("Paramater Inputan")

def input_user():
    sepal_length = st.sidebar.slider("Sepal Length", 0.1, 10.0, 0.5)
    sepal_width = st.sidebar.slider("Sepal Width", 0.1, 10.0, 2.1)
    petal_length = st.sidebar.slider("Petal Length", 0.1, 10.0, 1.0)
    petal_width = st.sidebar.slider("Petal Width", 0.1, 10.0, 0.1)
    data = {'sepal_length' : sepal_length,
            'sepal_width' : sepal_width,
            'petal_length' : petal_length,
            'petal_width' : petal_width}
    fitur = pd.DataFrame(data, index=[0])
    return fitur

df = input_user()

st.subheader('Parameter Inputan')
st.write(df)

load_clf = pickle.load(open('knn.pkl', 'rb'))

prediksi = load_clf.predict(df)
prediksi_proba = load_clf.predict_proba(df)

species = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
st.write(species[prediksi])
st.write(prediksi_proba)