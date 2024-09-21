import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV for face detection (using Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize OpenCV for capturing video
cap = cv2.VideoCapture(0)

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

# Function to find the closest face to a hand
def find_closest_face(hand_landmarks, faces):
    hand_x = hand_landmarks.landmark[0].x  # Hand center point
    hand_y = hand_landmarks.landmark[0].y
    closest_face = None
    min_distance = float('inf')

    for (x, y, w, h) in faces:
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        distance = (face_center_x - hand_x)**2 + (face_center_y - hand_y)**2

        if distance < min_distance:
            min_distance = distance
            closest_face = (x, y, w, h)

    return closest_face

# Main loop for real-time hand and face detection
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=4) as hands:
    max_count=0
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

        # To track how many people have raised only one hand
        hands_per_face = {}  # Dictionary to track hands per face

        if result.multi_handedness and result.multi_hand_landmarks:
            # Iterate through the detected hands
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Check if the hand is open
                if is_hand_open(hand_landmarks):
                    # Find the closest face to the detected hand
                    closest_face = find_closest_face(hand_landmarks, faces)

                    if closest_face:
                        face_key = tuple(closest_face)  # Use face coordinates as a unique identifier for a person
                        
                        # Track hands for each face
                        if face_key not in hands_per_face:
                            hands_per_face[face_key] = 1  # First hand detected for this face
                        else:
                            hands_per_face[face_key] += 1  # Second hand detected for this face

                        # Draw hand landmarks on the frame
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Count people who raised at least one hand
        people_with_hand_raised = len(hands_per_face)
        if people_with_hand_raised > max_count:
            max_count=people_with_hand_raised 

        # Display the count of people who raised at least one hand
        cv2.putText(frame, f'People with Hand Raised: {people_with_hand_raised}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Max Count: {max_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the frame with hand landmarks and face detection
        cv2.imshow('Real-Time Hand Detection with Face Tracking', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()