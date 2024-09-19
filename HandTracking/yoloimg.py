import torch
import cv2
import mediapipe as mp

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Mediapipe initialization for detecting keypoints (like hands)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_hands_raised(image_path):
    # Load image
    img = cv2.imread(image_path)
    results = model(img)

    # Parse YOLOv5 detections
    detections = results.xyxy[0]  # Bounding boxes and scores

    # Loop over all detections
    for *box, conf, cls in detections:
        if int(cls) == 0:  # YOLO class '0' is for persons
            x1, y1, x2, y2 = map(int, box)  # Person's bounding box

            # Extract the person from the image
            person_img = img[y1:y2, x1:x2]

            # Use Mediapipe Pose to detect keypoints
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            results = pose.process(person_rgb)

            # Check if hands are detected above the head
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract key points for hand and head
                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                nose = landmarks[mp_pose.PoseLandmark.NOSE]

                # Scale back to original image size
                img_h, img_w, _ = person_img.shape
                left_hand_y = left_hand.y * img_h
                right_hand_y = right_hand.y * img_h
                nose_y = nose.y * img_h

                # Determine if either hand is raised above the head (nose level)
                if left_hand_y < nose_y or right_hand_y < nose_y:
                    print(f"Person at [{x1}, {y1}, {x2}, {y2}] has raised their hands!")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    print(f"Person at [{x1}, {y1}, {x2}, {y2}] has not raised their hands.")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display result
    cv2.imshow("Hands Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_hands_raised('hands.jpeg')
