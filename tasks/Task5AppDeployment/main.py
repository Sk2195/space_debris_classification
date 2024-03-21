#!/usr/bin/env python
# coding: utf-8

# In[16]:

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
import warnings


# Disable this warning by disabling the config option:

st.set_option('deprecation.showPyplotGlobalUse', False)

# Set Streamlit page configuration
st.set_page_config(page_title='Classifying Space Debris', page_icon='SD', initial_sidebar_state="expanded")

# Set sidebar CSS style
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content .block-container {
        color: #000000;
    }
    .sidebar .sidebar-content .block-container .block {
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content .block-container .sidebar-title {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide Streamlit default elements
hide_streamlit_style = """
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load GIF assets
gif_2 = Image.open("tasks/Task5AppDeployment/assets/gifs/debris2.gif")
gif_3 = Image.open("tasks/Task5AppDeployment/assets/gifs/debris3.gif")


def define_homepage():
    st.write("# Space Debris Classifier")
    st.image(gif_2)
    st.write("""
    ## What is space debris?
    - Space Debris, also known as space junk, is a form of debris left by humans in space. This term may be applied to any large objects and dead satellites that remain in orbit after their purpose is over. Additionally, it can also be applied to any smaller objects, paint
    flecks, or pieces that might have fallen off a rocket.
    
   ## Does Space Debris pose a serious threat to satellites? 
   -  Considering that these objects travel at a speed of 25,000 kilometers per hour, even the tiniest debris can be a serious threat to satellites, space missions, and cause significant damage. Every fragment of space junk can be a barrier to the orbital highway, increasing the risk of collision for functioning satellites.
   
 # What actions are being taken by the U.S. Government?
 - As of five months ago, the FCC took action against DISH Network to pay $150,000 for its failure to move a satellite to a safer orbit. While the amount might be small, this action from the U.S. government gives us a sneak peek into the future of an ever-growing problem. For more news on this, please check the link here: [FCC Issues First Space Debris Fine](https://www.linkedin.com/news/story/fcc-issues-first-space-debris-fine-5503617/)
    
 ### Why did I create this model?
 -  It is imperative that as this crisis grows, we are able to use AI and the power of data science to create a model that can classify space debris. By using cutting-edge Machine Learning techniques and evaluating sensor data, we can aid space agencies to categorize debris fragments based on their characteristics. Additionally, making a model that can categorize space debris can help satellites avoid collision and plan our space missions better.
""")
    st.write("""
             ## What are the project goals and where is the dataset from?
             - The objective of this project is to create a model that accurately classifies space debris based on several characteristics. Additionally, the other two objectives are to provide an advance warning to satellite
             operator so they can make better decisions to avoid collisions and promote the longevity of space missions.
           
 - The data we used for this model is originally from SpaceTrack.Org. This dataset was directly downloaded from this link: [Space Debris Dataset](https://www.kaggle.com/code/kandhalkhandeka/predicting-rcs-size-of-space-debris).
   ## Is there an API available or can the data be downloaded directly from Space Track?
 - There is also a public API that allows users to extract data directly to their local drive.In terms of myself, I was able to extract the data using the API provided. However, I am using the data avaliable on Kaggle due to confusion in data usage policy.
    """)

def about_page():

    st.write("""
    ## Project Description
     - The Space Debris Classifier projects aims to develop a cutting-edge maching learning technique to identify space junk. With the through analysis of several characteristics like RCS size, inclination, eccentricity etc.. the likelihood of collison can promote a more sustainable mission and safe operations.
 """)
    st.write(""" 
    ## Project Variable Description Used
     - MEAN_MOTION represents average angular velocity of a satellite 
     - ECCENTRICITY : represents how 'circular' the object is.
     - INCLINATION : an angle that lies between the satellite orbital plan and celestial body it orbits.
     - RA_OF_ASC_NODE : specifies the orbital of an object in space by pinpointing the specified direction
     - ARG_OF_PERICENTER: an angle of a orbital body that lies between the ascending node to its periapsis
     - REV_AT_EPOCH: Number of complete orbits or revolutions a satellite made
     - BSTAR: coefficient linked to atmospheric drag
     - MEAN_MOTION_DOT: 1st derivative of mean_motion
     - MEAN_MOTION_DDOT: 2nd derivative of mean_motion
     - SEMIMAJOR_AXIS: longest diameter
     - PERIOD : Orbital period of the satellite
     - APOPAPSIS: Farthest point from an satellite
     - PERIAPSIS: Nearest point from the satellite
     - AP_DIFF: Difference between the APOPAPSIS and PERIAPSIS
     - SEMIMINOR_AXIS: shortest diameter
     - ORBITAL_VELOCITY: speed of a satellite
     - TARGET : Represents Space Debris Classification as either 0 and 1
    """)
    
def model_selection():
    model_path = "tasks/Task5AppDeployment/assets/model.pkl"  # Update with your model path
    model = joblib.load(model_path)

    # Set the title and the description of the app
    st.title('Space Debris Classification')
    st.write('This app predicts whether an object is space debris or not based on the provided features.')

    # Create input elements for the user input
    input_elements = {}
    feature_names = ['MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE',
                     'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'REV_AT_EPOCH', 'BSTAR',
                     'MEAN_MOTION_DOT', 'MEAN_MOTION_DDOT', 'SEMIMAJOR_AXIS', 'PERIOD',
                     'APOAPSIS', 'PERIAPSIS', 'AP_DIFF', 'SEMIMINOR_AXIS',
                     'ORBITAL_VELOCITY', 'RCS_SIZE_MEDIUM', 'RCS_SIZE_SMALL']

    # Define the ranges based on the analysis
    ranges = {
        'MEAN_MOTION': (0.5, 17.5),
        'ECCENTRICITY': (0, 0.3),
        'INCLINATION': (0, 180),
        'RA_OF_ASC_NODE': (0, 360),
        'ARG_OF_PERICENTER': (0, 360),
        'MEAN_ANOMALY': (0, 360),
        'REV_AT_EPOCH': (0, 100000),
        'BSTAR': (0, 1),
        'MEAN_MOTION_DOT': (-0.015, 0.225),  
        'MEAN_MOTION_DDOT': (-0.000042, 0.00165), 
        'SEMIMAJOR_AXIS': (6500, 305000),  
        'PERIOD': (87, 27810),
        'APOAPSIS': (180, 372140),
        'PERIAPSIS': (70, 224550),
        'AP_DIFF': (0, 292240),  
        'SEMIMINOR_AXIS': (6500, 295130),  
        'ORBITAL_VELOCITY': (1, 8)  
    }
    for feature in feature_names:
        if feature in ['RCS_SIZE_MEDIUM', 'RCS_SIZE_SMALL']:
            input_elements[feature] = st.selectbox(feature, [0, 1])
        else:
            min_val, max_val = ranges[feature]
            min_val = float(min_val)  # Ensure consistency in data types
            max_val = float(max_val)
            default_val = (min_val + max_val) / 2.0
            input_elements[feature] = st.slider(feature, min_val, max_val, default_val)


    # Create a button for prediction
    if st.button('Predict'):
        # Prepare the data
        input_data = [[input_elements[feature] for feature in feature_names]]

        # Make predictions using the model
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        # Display the predicted class and probability
        if predictions[0] == 0:
            st.write('Prediction: Not Space Debris')
        else:
            st.write('Prediction: Space Debris')
        st.write('Probability:', probabilities[0][1])

    # Optional: Calculate feature importances if your model supports it
    try:
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance[:len(feature_names)]
        })

        # Display feature importance
        st.subheader('Feature Importance')
        st.dataframe(feature_importance)
    except AttributeError:
        st.write("Feature importances are not available for the selected model.")


def visualization_page():
    st.write("""
    # Space Debris Visualization
    This page displays visualizations of space debris data.
    """)

    # Load the cleaned space data
    cleaned_spacedata_path = r"data/cleaned data/spacedebris_clean2.csv"
    
    cleaned_spacedata = pd.read_csv(cleaned_spacedata_path)

    # Display the summary statistics
    st.write('Summary Statistics')
    st.write(cleaned_spacedata.describe())

    # Object Type in Space
    # Count the occurrences of each object type
    object_type_counts = cleaned_spacedata['OBJECT_TYPE'].value_counts()

    # Plot the distribution of object types
    plt.figure(figsize=(10, 6))
    object_type_counts.plot(kind='bar')
    plt.xlabel('Object Type')
    plt.ylabel('Count')
    plt.title('Distribution of Object Types')
    plt.xticks(rotation=45)
    st.pyplot()

    # RCS Size Distribution
    # Count the occurrences of each RCS size
    rcs_size_counts = cleaned_spacedata['RCS_SIZE'].value_counts()

    # Plot the distribution of RCS sizes
    plt.figure(figsize=(10, 6))
    rcs_size_counts.plot(kind='bar')
    plt.xlabel('RCS Size')
    plt.ylabel('Count')
    plt.title('Distribution of RCS Sizes')
    plt.xticks(rotation=45)
    st.pyplot()

    # Crosstab of Object Type and RCS Size
    cross_table = pd.crosstab(cleaned_spacedata['OBJECT_TYPE'], cleaned_spacedata['RCS_SIZE'])

    # Plot the stacked bar chart
    plt.figure(figsize=(10, 6))
    cross_table.plot(kind='bar', stacked=True)
    plt.xlabel('Object Type')
    plt.ylabel('Count')
    plt.title('Distribution of RCS Size by Object Type')
    plt.xticks(rotation=45)
    plt.legend(title='RCS Size')
    st.pyplot()

    # Correlation Plot
    correlation_data_path = r"data/pre-processed datas-final"
    correlation_data = pd.read_csv(correlation_data_path)

    plt.figure(figsize=(10, 6))
    correlation_matrix = correlation_data.corr()
    st.write('Correlation Matrix:')
    st.dataframe(correlation_matrix)
    st.write('Heatmap:')
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot()


def main():
    with st.sidebar:
        st.image(gif_3)
        st.sidebar.title('')  # Remove the title from the sidebar

    selected = st.sidebar.selectbox(
        '',
        ['Home', 'About', 'Visualization'] + ['Classify Space Measurement'],
        format_func=lambda x: x.split()[-1] if x != 'Home' else x
    )

    if selected == 'Home':
        define_homepage()
    elif selected == 'About':
        about_page()
    elif selected == 'Classify Space Measurement':
        model_selection()
    elif selected == 'Visualization':
        visualization_page()


if __name__ == '__main__':
    main()
