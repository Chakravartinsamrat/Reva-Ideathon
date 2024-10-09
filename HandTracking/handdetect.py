import cv2
import mediapipe as mp

# Load the image
image_path = 'hands.jpeg'  # Replace with your image path
image = cv2.imread(image_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Convert the image to RGB for MediaPipe
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image with MediaPipe Pose for multiple people
results = pose.process(image_rgb)

# Initialize the list to store detected landmarks for multiple people
people_landmarks = []

# Initialize hand raised count for each detected person
raised_hands_count = 0

# Check if any pose landmarks were detected
if results.pose_landmarks:
    for i, pose_landmarks in enumerate(results.pose_landmarks):
        # Extract the landmarks for each detected person
        landmarks = pose_landmarks.landmark
        
        # Store landmarks for tracking multiple people
        people_landmarks.append(landmarks)
        
        # Extract keypoints for head (nose) and wrists for this person
        head_y = landmarks[0].y  # Nose (head)
        left_wrist_y = landmarks[15].y  # Left wrist
        right_wrist_y = landmarks[16].y  # Right wrist

        # Determine if hands are raised above head
        hands_raised = (left_wrist_y < head_y) or (right_wrist_y < head_y)
        raised_hands_count += 1 if hands_raised else 0
        
        # Annotate each person's pose in the image
        mp_drawing.draw_landmarks(
            image, 
            pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Landmarks style
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)   # Connections style
        )

# Save or display the image with drawn pose landmarks
output_image_path = 'output_multi_people_pose_image.jpg'
cv2.imwrite(output_image_path, image)

# Display the result
cv2.imshow("Multiple People Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Total raised hands detected across all people: {raised_hands_count}")
