# Import necessary libraries
import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Title of the app
st.title("YOLOv8 Hand Detection")

# Load the trained YOLO model
model = YOLO('best.pt')

# File uploader to upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_image)
    img_array = np.array(image)

    # Make predictions
    results = model(img_array)

    # Draw bounding boxes on the image without labels
    annotated_image = results[0].plot(conf=False, labels=False)

    # Display the image with bounding boxes
    st.image(annotated_image, caption="Detected Hands", use_column_width=True)

    # Extract class-wise count from predictions
    hand_count = sum([1 for r in results[0].boxes.cls if r == 0])  # Class 0: Hand
    maybe_hand_count = sum([1 for r in results[0].boxes.cls if r == 1])  # Class 1: Maybe Hand
    no_hand_count = sum([1 for r in results[0].boxes.cls if r == 2])  # Class 2: No Hand

    # Display the detection summary
    st.write(f"0: {results[0].orig_shape[0]}x{results[0].orig_shape[1]} {hand_count} Hand(s), "
             f"{maybe_hand_count} Maybe Hand(s), {no_hand_count} No Hand(s), {results[0].speed['inference']:.1f}ms")
    
    # Display speed details
    st.write(f"Speed: {results[0].speed['preprocess']:.1f}ms preprocess, "
             f"{results[0].speed['inference']:.1f}ms inference, "
             f"{results[0].speed['postprocess']:.1f}ms postprocess per image at shape {results[0].orig_shape}")
