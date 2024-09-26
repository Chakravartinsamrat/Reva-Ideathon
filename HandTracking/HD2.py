import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV for face detection (using Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize OpenCV for capturing video
cap = cv2.VideoCapture(0)

# Set frame width and height (increase frame size)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height to 720 pixels

# Function to determine if a hand is open or closed
def is_hand_open(hand_landmarks):
    # Compare the tip of the fingers with their respective MCP joints
    finger_tips = [4, 8, 12, 16, 20]  # Index for thumb, index, middle, ring, pinky
    finger_open = []

    for tip in finger_tips[1:]:  # Ignoring thumb for simplicity
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:  # Tip should be higher than MCP
            finger_open.append(True)
        else:
            finger_open.append(False)

    return sum(finger_open) >= 3  # At least 3 fingers should be open to consider it a raised hand

# Main loop for real-time hand and face detection
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=4) as hands:
    max_count = 0  # Initialize max count of hands raised
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a natural selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the image to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Convert the image to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        result = hands.process(rgb_frame)

        # To track raised hands in real-time
        raised_hand_count = 0

        # If hands are detected, check if they are open and draw landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if is_hand_open(hand_landmarks):
                    raised_hand_count += 1  # Increment count for each raised hand
                    # Draw hand landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Update max count if necessary
        if raised_hand_count > max_count:
            max_count = raised_hand_count

        # Display the current raised hand count, the maximum count, and the face count
        face_count = len(faces)  # Count the number of detected faces
        cv2.putText(frame, f'Current Hand Raised Count: {raised_hand_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(frame, f'Max Hand Raised Count: {max_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(frame, f'Face Count: {face_count}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        # Show the frame with hand landmarks and face detection
        cv2.imshow('Real-Time Hand and Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()