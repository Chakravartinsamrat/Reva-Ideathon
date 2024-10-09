import streamlit as st
# import torch
from ultralytics import YOLO
import cv2

# st.title("Hand Raised Detection")
# if __name__ == "__main__":
#     model = YOLO('yolov8n.pt')  # Load your YOLOv8 model
#     model.train(data='/Users/piyus/Documents/GitHub/HandSuP/dataset2/data.yaml',
#                 epochs=50,
#                 batch=8,
#                 imgsz=640,
#                 device='cuda')
model = YOLO('./yolo_models/best.pt') 
results = model('./test_images/test4.jpg')


# Loop through each detection result and draw a green bounding box
# for result in results:
#     img = result.orig_img  # Original image
    
#     for box in result.boxes:  # Iterate through detected boxes
#         # Get bounding box coordinates
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
        
#         # Draw a green rectangle around the detected hand
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box (0, 255, 0)
    
#     # Save or display the image
#     cv2.imwrite('output.jpg', img)  # Save the result
#     cv2.imshow('Detected Image', img)  # Show the image with detection
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()