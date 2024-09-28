# Required Libraries
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# Function to detect faces, mark them, and display the face count with zoom ability
def detect_and_mark_faces(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (required for MTCNN)

    # Initialize the MTCNN face detector
    detector = MTCNN()

    # Detect faces
    faces = detector.detect_faces(img_rgb)

    # Get face count
    face_count = len(faces)
    print(f"Number of faces detected: {face_count}")

    # Draw bounding boxes around the detected faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Add text for the face count at the top left corner of the image
    cv2.putText(img_rgb, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the image with marked faces and zoom ability
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f'Detected Faces: {face_count}')
    plt.axis('off')

    # Enable interactive zooming using matplotlib's functionality
    plt.gca().set_title(f'Detected Faces: {face_count}', fontsize=14)
    plt.show()

# Example usage
#image_path = 'SAM/image2.png' 
#image_path = 'SAM/image2.png'
image_path = 'Reva-Ideathon/darshan/src/image3.png' 
detect_and_mark_faces(image_path)