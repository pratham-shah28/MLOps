import streamlit as st
import requests
import os

st.title('Wine Class Prediction')

# 13 numeric feature inputs for Wine dataset
alcohol = st.number_input('Alcohol', min_value=0.0, max_value=20.0, step=0.1)
malic_acid = st.number_input('Malic Acid', min_value=0.0, max_value=10.0, step=0.1)
ash = st.number_input('Ash', min_value=0.0, max_value=10.0, step=0.1)
alcalinity_of_ash = st.number_input('Alcalinity of Ash', min_value=0.0, max_value=50.0, step=0.1)
magnesium = st.number_input('Magnesium', min_value=0.0, max_value=200.0, step=1.0)
total_phenols = st.number_input('Total Phenols', min_value=0.0, max_value=10.0, step=0.1)
flavanoids = st.number_input('Flavanoids', min_value=0.0, max_value=10.0, step=0.1)
nonflavanoid_phenols = st.number_input('Nonflavanoid Phenols', min_value=0.0, max_value=5.0, step=0.1)
proanthocyanins = st.number_input('Proanthocyanins', min_value=0.0, max_value=5.0, step=0.1)
color_intensity = st.number_input('Color Intensity', min_value=0.0, max_value=20.0, step=0.1)
hue = st.number_input('Hue', min_value=0.0, max_value=2.0, step=0.1)
od_ratio = st.number_input('OD280/OD315 of Diluted Wines', min_value=0.0, max_value=5.0, step=0.1)
proline = st.number_input('Proline', min_value=0.0, max_value=2000.0, step=10.0)

if st.button('Predict'):
    data = {
        'alcohol': alcohol,
        'malic_acid': malic_acid,
        'ash': ash,
        'alcalinity_of_ash': alcalinity_of_ash,
        'magnesium': magnesium,
        'total_phenols': total_phenols,
        'flavanoids': flavanoids,
        'nonflavanoid_phenols': nonflavanoid_phenols,
        'proanthocyanins': proanthocyanins,
        'color_intensity': color_intensity,
        'hue': hue,
        'od280/od315_of_diluted_wines': od_ratio,
        'proline': proline
    }
    try:
        response = requests.post('https://wine-app-797035919127.us-east4.run.app/predict', json=data)
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f'Predicted Wine Class: {prediction}')
        else:
            st.error(f'Error occurred during prediction. Status code: {response.status_code}')
    except requests.exceptions.RequestException as e:
        st.error(f'Error occurred during prediction: {str(e)}')

# temp
