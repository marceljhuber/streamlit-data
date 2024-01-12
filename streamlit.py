import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time

d1, d2, d3 = np.load("train_w.npy"), np.load("test_w.npy"), np.load("pred_w.npy")
d1, d2 = d1[-163:], d2[-163:]

tab1, tab2 = st.tabs(["Image Processing", "Training and Dataframe"])

with tab1:
    st.header("Image Manipulation")

    col1, col2 = st.columns([.7, .3], gap="medium")
    new_image = None

    with col1:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:

            img = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            col11, col12 = st.columns(2, gap="small")
            with col11:
                color = st.radio(
                    'Color Type',
                    ("Normal", "Reversed", "Gray-scale"))
            with col12:
                flip = st.radio(
                    'Flip Type',
                    ("None", "Vertical", "Horizontal"))

            values = st.slider('Select the opacity rate', 0, 100, 100)

            if st.button("Apply"):
                if color == "Reversed":
                    new_image = 255 - img
                elif color == "Gray-scale":
                    new_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    new_image = img

                if flip == "Vertical":
                    new_image = cv2.flip(new_image, 0)
                elif flip == "Horizontal":
                    new_image = cv2.flip(new_image, 1)

                if color != "Gray-scale":
                    new_image = np.concatenate((new_image,
                                                np.ones((new_image.shape[0], new_image.shape[1], 1), dtype=int) * int(
                                                    values * 2.55)), axis=-1)

        with col2:
            if uploaded_file is not None:
                st.image(img, caption='Uploaded Image.', use_column_width=False, width=175)
                st.write("Image shape:", img.shape)
                if new_image is not None:
                    st.image(new_image, "Manipulated Image", use_column_width=False, width=175)

with tab2:
    st.subheader("Training LSTM with Tabular Data", divider='rainbow')
    df = pd.read_csv("./formatted.csv")
    "This is the temperature log in celsius degree in specific time range"
    st.dataframe(df, use_container_width=True)

    st.caption('Before you start, please start the training')
    if st.button("Start Training"):
        with st.spinner('Data is preprocessing...'):
            time.sleep(1.3)
        with st.spinner('Training...'):
            time.sleep(2.2)
        with st.spinner('Isolating the Predictions...'):
            time.sleep(1.2)
        st.success('Done!', icon="✅")

    "\n"
    "This is the training data, the LSTM network is fed with this data with sliciding window size of 5"
    st.line_chart(d1, use_container_width=True)

    "\n\n"
    st.subheader('Visualizing the Results', divider='rainbow')

    plot = st.radio(
        "Please select which datas you want to plot",
        ["Test Data", "Prediction Data", "Test and Prediction Data"], index=0, horizontal=True)

    if plot == "Test and Prediction Data":
        ddf = pd.DataFrame(np.concatenate((d2[:, np.newaxis], d3[:, np.newaxis]), axis=-1), columns=["Test", "Pred"])

    elif plot == "Test Data":
        ddf = pd.DataFrame(d2, columns=['Test'])
    else:
        ddf = pd.DataFrame(d3, columns=['Pred'])

    st.line_chart(ddf)

    st.divider()
    st.header('Thank you for participating!')
    st.link_button("Hit this button for more and to keep up-to-date with our events!!!", "https://neuron-ai.at/")
    st.divider()
