import streamlit as st
import tensorflow as tf
import numpy as np

# Fruit/vegetable nutritional info
nutrition_info = {
    'apple': 'Calories: 52, Carbs: 14g, Protein: 0.3g, Fat: 0.2g',
    'banana': 'Calories: 96, Carbs: 27g, Protein: 1.3g, Fat: 0.3g',
    'beetroot': 'Calories: 43, Carbs: 9.6g, Protein: 1.6g, Fat: 0.2g',
    'bell pepper': 'Calories: 20, Carbs: 4.7g, Protein: 0.9g, Fat: 0.2g',
    'cabbage': 'Calories: 25, Carbs: 5.8g, Protein: 1.3g, Fat: 0.1g',
    'capsicum': 'Calories: 20, Carbs: 4.7g, Protein: 0.9g, Fat: 0.2g',
    'carrot': 'Calories: 41, Carbs: 10g, Protein: 0.9g, Fat: 0.2g',
    'cauliflower': 'Calories: 25, Carbs: 5g, Protein: 2g, Fat: 0.3g',
    'chilli pepper': 'Calories: 40, Carbs: 9g, Protein: 2g, Fat: 0.4g',
    'corn': 'Calories: 96, Carbs: 21g, Protein: 3.4g, Fat: 1.5g',
    'cucumber': 'Calories: 15, Carbs: 3.6g, Protein: 0.7g, Fat: 0.1g',
    'eggplant': 'Calories: 25, Carbs: 5.9g, Protein: 0.8g, Fat: 0.2g',
    'garlic': 'Calories: 149, Carbs: 33g, Protein: 6.4g, Fat: 0.5g',
    'ginger': 'Calories: 80, Carbs: 18g, Protein: 1.8g, Fat: 0.8g',
    'grapes': 'Calories: 69, Carbs: 18g, Protein: 0.7g, Fat: 0.2g',
    'jalepeno': 'Calories: 29, Carbs: 6g, Protein: 1g, Fat: 0.4g',
    'kiwi': 'Calories: 41, Carbs: 10g, Protein: 0.8g, Fat: 0.4g',
    'lemon': 'Calories: 29, Carbs: 9g, Protein: 1g, Fat: 0.3g',
    'lettuce': 'Calories: 5, Carbs: 1g, Protein: 0.5g, Fat: 0.1g',
    'mango': 'Calories: 60, Carbs: 15g, Protein: 0.8g, Fat: 0.4g',
    'onion': 'Calories: 40, Carbs: 9g, Protein: 1g, Fat: 0.1g',
    'orange': 'Calories: 47, Carbs: 12g, Protein: 1g, Fat: 0.1g',
    'paprika': 'Calories: 20, Carbs: 4.5g, Protein: 1g, Fat: 0.5g',
    'pear': 'Calories: 57, Carbs: 15g, Protein: 0.4g, Fat: 0.1g',
    'peas': 'Calories: 81, Carbs: 14g, Protein: 5g, Fat: 0.4g',
    'pineapple': 'Calories: 50, Carbs: 13g, Protein: 0.5g, Fat: 0.1g',
    'pomegranate': 'Calories: 83, Carbs: 19g, Protein: 1.7g, Fat: 1.2g',
    'potato': 'Calories: 77, Carbs: 17g, Protein: 2g, Fat: 0.1g',
    'raddish': 'Calories: 16, Carbs: 3.4g, Protein: 0.7g, Fat: 0.1g',
    'soy beans': 'Calories: 173, Carbs: 15g, Protein: 17g, Fat: 9g',
    'spinach': 'Calories: 23, Carbs: 3.6g, Protein: 2.9g, Fat: 0.4g',
    'sweetcorn': 'Calories: 96, Carbs: 21g, Protein: 3.4g, Fat: 1.5g',
    'sweetpotato': 'Calories: 86, Carbs: 20g, Protein: 1.6g, Fat: 0.1g',
    'tomato': 'Calories: 18, Carbs: 3.9g, Protein: 0.9g, Fat: 0.2g',
    'turnip': 'Calories: 28, Carbs: 6.4g, Protein: 1g, Fat: 0.1g',
    'watermelon': 'Calories: 30, Carbs: 8g, Protein: 0.6g, Fat: 0.2g'
}

st.set_page_config(page_title="What is this Fruit or Vegetable?", layout="wide")
st.title('🍌🍓 What is this Fruit or Vegetable? 🥒🌽')
st.write("Upload an image of a fruit or vegetable, and I'll tell you what it is and give you some nutritional information!")

cols = st.columns(3)
with cols[1]:
    uploaded_file = st.file_uploader(label="", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Load the model (place this in /model/ directory in GitHub)
model = tf.keras.models.load_model('model/WhatIsThisFruit.keras')

data_categories = list(nutrition_info.keys())

if uploaded_file is not None:
    image = tf.keras.utils.load_img(uploaded_file, target_size=(180, 180))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])
    predicted_class = data_categories[np.argmax(score)]
    confidence = np.max(score) * 100

    img_cols = st.columns(3)
    with img_cols[1]:
        st.image(image, width=200)

    st.markdown(f"**Prediction:** {predicted_class} with **confidence** {confidence:0.2f}%")
    st.markdown(f"**Nutritional Information:** {nutrition_info.get(predicted_class, 'Not available')}")

# CSS styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600&display=swap');
        html, body, .stApp {
            font-family: 'Instrument Sans', sans-serif;
            background-color: #fff;
            text-align: center;
        }
        .stApp {
            margin-top: 50px;
        }
        .stFileUploader {
            background-color: #BFBFBF;
            border-radius: 5px;
            padding: 5px;
            margin-bottom: 10px;
        }
        .stFileUploader button {
            background-color: #BFBFBF;
            border: none;
            border-radius: 6px;
            padding: 8px;
            color: black;
            font-weight: bold;
        }
        img {
            max-width: 300px;
            height: auto;
            border-radius: 12px;
            margin-top: 20px;
            margin-left: 120px;
        }
    </style>
""", unsafe_allow_html=True)
