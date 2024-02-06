import numpy as np
import streamlit as st
import re
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.write("""
<div style='text-align:center'>
    <h1 style='color:#009999;'>PREDICTING WHO ARE ALL COMING AS VISITORS AND CONVERTED AS OUT CUSTOMERS(CLASSIFICATION | PREDICTION</h1>
</div>
""", unsafe_allow_html=True)

selected_tab = st.selectbox("Select Tab", ["PREDICT STATUS"])

if selected_tab == "PREDICT STATUS":
    geoNetwork_region = ['Sharjah', 'Ajman', 'Abu Dhabi', 'Dubai', 'Ras al Khaimah', 'England', 'Hessen', 'Ash Sharqia Governorate', 'Nabatiyeh Governorate', 'Umm Al Quawain', 'Indiana', 'Illinois',
                         'Makkah Province', 'Maharashtra', 'Capital Governorate', 'Jakarta', 'Punjab', 'Mount Lebanon Governorate', 'South Holland', 'Riyadh Province', 'Ile-de-France', 'Chandigarh', 'Karnataka',
                         'Istanbul', 'Cairo Governorate', 'Andhra Pradesh', 'Amman Governorate', 'Beirut Governorate', 'Alexandria Governorate', 'New York', 'Ontario', 'Davao Region', 'Auvergne-Rhone-Alpes', 'Fujairah',
                         'Tamil Nadu', 'Pays de la Loire', 'Metro Manila', 'Normandy', 'Giza Governorate', 'Bavaria', 'Stockholm County', 'Tel Aviv District', 'Vienna', 'Decentralized Administration of Peloponnese',
                         'Western Greece and the Ionian', 'Porto District', 'West Bengal', 'Kerala', 'Lombardy', 'MIMAROPA', 'North Holland', 'Chiba', 'Islamabad Capital Territory', 'Ohio', 'Florida', 'Moscow', 'County Dublin',
                         'Quebec', 'Muscat Governorate', 'Telangana', 'Vastra Gotaland County', 'Western Cape', 'North Rhine-Westphalia', 'Sindh', 'Virginia', 'Calabarzon', 'Shanghai', 'Central Luzon', 'Dakahlia Governorate',
                         'Western Province', 'Flanders', 'Fars', 'Washington', 'Tashkent Region', 'Eastern Province', 'Federal Territory of Kuala Lumpur', 'Goa', 'Southern Governorate', 'Madhya Pradesh', 'Ismailia Governorate',
                         'Uttar Pradesh', 'Casablanca-Settat', 'Nairobi County', 'Sofala Province', 'Attica', 'Tokyo', 'Khyber Pakhtunkhwa', 'Gujarat', 'Assam', 'Brittany', 'Centre-Val de Loire']

    with st.form('my_form'):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            count_session = st.text_input('Enter count session (Min:1 & Max:270)')
            count_hit = st.text_input('Enter count hit (Min:2 & Max:48744)')
            avg_session_time = st.text_input('Enter avg session time(Min:2.0, Max:5441.0)')
            sessionQualityDim = st.text_input('sessionQualityDim (Min:1, Max:100)')
            
        with col3:
            visits_per_day = st.text_input('visits_per_day (Min:0.9230769230769232, Max:2397.333333333333)')
            historic_session = st.text_input('historic_session (Min:2, Max:181715)')
            selected_region = st.selectbox('geoNetwork_region', geoNetwork_region, key=1)

            submit_button = st.form_submit_button(label='PREDICT STATUS')
            st.markdown('''
            ''', unsafe_allow_html=True)

        flag = 0
        pattern = '^(?:\d+|\d*\.\d+)$'
        for i in [count_session, count_hit, avg_session_time, sessionQualityDim, visits_per_day, historic_session]:
            if re.match(pattern, i):
                pass
            else:
                flag = 1
                break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write('Please enter a valid number. Spaces are not allowed.')
        else:
            st.write('You have entered an invalid value: ', i)

    if submit_button and flag == 0:
        # Load the objects from pickle files
        with open(r"cmodel.pkl", 'rb') as file:
            cloaded_model = pickle.load(file)

        with open(r'cscaler.pkl', 'rb') as f:
            cscaler_loaded = pickle.load(f)

        with open(r"ct.pkl", 'rb') as f:
            ct_loaded = pickle.load(f)

        # Assuming new_sample is defined and contains the new sample data
        new_sample = np.array([[np.log(float(count_session)), np.log(float(count_hit)), np.log(float(avg_session_time)), float(sessionQualityDim), int(visits_per_day), selected_region, int(historic_session)]])
        # Transform new sample
        new_sample_encoded = ct_loaded.transform(new_sample[:, [5]])  # Selecting the sixth column directly
        new_sample_encoded = pd.DataFrame(new_sample_encoded.toarray(), columns=ct_loaded.get_feature_names_out(['geoNetwork_region']))
        new_sample_array = new_sample[:, [0, 1, 2, 3, 4]]  # Selecting the numeric columns only
        new_sample = np.concatenate((new_sample_array, new_sample_encoded), axis=1)

        new_sample_scaled = cscaler_loaded.transform(new_sample)

        # Predict using RandomForestClassifier
        new_pred = cloaded_model.predict(new_sample_scaled)

        # Output the prediction
        if 1 in new_pred:
            st.write('The status is: converted')
        else:
            st.write('The status is: Not converted')
