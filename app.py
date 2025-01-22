import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

model = YOLO(r'C:\Users\Nasem\Desktop\muzzledetector\best.pt')

# Streamlit app
st.title("Muzzle detection with YOLOv8")
st.write("Upload an image, and the YOLOv8 model will detect objects and display the results.")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Run YOLOv8 inference
    results = model.predict(image_np, save=False, conf=0.5, iou=0.7) 
    result_image = results[0].plot()  # Get the annotated image directly from YOLOv8 results

    # Display original and annotated images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.header("Detection Result")
        st.image(result_image, caption="YOLOv8 Detection", use_column_width=True)
