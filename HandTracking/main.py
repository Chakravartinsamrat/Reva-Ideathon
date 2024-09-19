import os
import cv2
import mediapipe as mp
import time

# Disable a TensorFlow warning (if applicable)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # some random warning from tensorflow

# Initialize video capture from camera 0
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model, allowing detection of up to 10 hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=10)  # Allow detection of up to 10 hands
mpDraw = mp.solutions.drawing_utils  # Utility to draw landmarks on hands

# Variables to track the count of raised hands and the highest count observed
raised_count = 0
highest_count = 0

# Function to check if a hand is "raised" based on its wrist and landmarks
def is_hand_raised(landmarks):
    # Wrist is landmark 0, get its y-coordinate
    wrist_y = landmarks[0].y

    # Get the average y-coordinate of fingers (landmarks 5 to 17 represent fingers)
    finger_y_avg = sum([landmarks[i].y for i in range(5, 21)]) / 16

    # If wrist is lower than the average finger position (higher y value in the image), it's "raised"
    return wrist_y > finger_y_avg

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB as required by MediaPipe Hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # List to store if a hand is raised or not
    current_raised_hands = []

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Check if the hand is raised
            if is_hand_raised(handLms.landmark):
                current_raised_hands.append(True)
            else:
                current_raised_hands.append(False)

            # Draw hand landmarks on the image for visualization
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate the current count of raised hands
    new_raised_count = sum(current_raised_hands)

    # Update highest count if needed
    if new_raised_count > highest_count:
        highest_count = new_raised_count

    # Display the counts on the image
    cv2.putText(img, f"Raised Hands: {new_raised_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f"Highest Count: {highest_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed image
    cv2.imshow("Hand Tracker", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()
