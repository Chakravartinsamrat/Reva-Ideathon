import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import numpy as np

# Function to detect faces, mark them, and display the face count
def detect_and_mark_faces(image):
    # Convert the image to RGB format (required for MTCNN)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize the MTCNN face detector
    detector = MTCNN()

    # Detect faces
    faces = detector.detect_faces(img_rgb)

    # Get face count
    face_count = len(faces)
    st.write(f"Number of faces detected: {face_count}")

    # Draw bounding boxes around the detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Add text for the face count at the top left corner of the image
    cv2.putText(img_rgb, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return img_rgb, face_count

# Streamlit app structure
st.title("Face Count App")
st.write("Upload an image (JPEG, PNG, etc.) and click 'Count Faces' to detect faces.")

# Image upload option
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# When an image is uploaded
if uploaded_file is not None:
    # Convert the uploaded image to OpenCV format
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Count button to trigger face detection
    if st.button("Count Faces"):
        # Detect and mark faces
        result_img, face_count = detect_and_mark_faces(image)

        # Display the result image with face count and bounding boxes
        st.image(result_img, caption=f"Detected Faces: {face_count}", use_column_width=True)