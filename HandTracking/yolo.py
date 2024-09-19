import torch
import cv2
import mediapipe as mp

# Load YOLOv5 model for person detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Mediapipe 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def detect_hands_raised_in_video(video_path, scale=1.5):
    # Capture video from file or webcam
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames are available
        
        frame= cv2.resize(frame,None,fx=scale,fy=scale)
        results = model(frame)

        detections = results.xyxy[0]

        # Loop over all detections in the current frame
        for *box, conf, cls in detections:
            if int(cls) == 0:  # Class 0 corresponds to 'person'
                x1, y1, x2, y2 = map(int, box)  # Person's bounding box

                # Extract the person from the frame
                person_img = frame[y1:y2, x1:x2]

                
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                keypoints_results = pose.process(person_rgb)

                
                if keypoints_results.pose_landmarks:
                    landmarks = keypoints_results.pose_landmarks.landmark

                    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                    nose = landmarks[mp_pose.PoseLandmark.NOSE]

                    img_h, img_w, _ = person_img.shape
                    left_hand_y = left_hand.y * img_h
                    right_hand_y = right_hand.y * img_h
                    nose_y = nose.y * img_h

                    
                    if left_hand_y < nose_y or right_hand_y < nose_y:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for raised hands
                        cv2.putText(frame, 'Hands Raised', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for no raised hands
                        cv2.putText(frame, 'Hands Not Raised', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame with bounding boxes and labels
        cv2.imshow('Hands Raised Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage with a video file:
#detect_hands_raised_in_video('children_raising_hands.mp4')

detect_hands_raised_in_video(0) 
