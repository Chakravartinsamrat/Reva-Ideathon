import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("children_raising_hands.mp4")

def is_hand_open(hand_landmarks):
    
    finger_tips = [4, 8, 12, 16, 20]  # Index for thumb, index, middle, ring, pinky
    finger_open = []

    for tip in finger_tips[1:]:  # Ignoring tumb
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:  # Check if the tip is higher than MCP
            finger_open.append(True)
        else:
            finger_open.append(False)

    
    return sum(finger_open) >= 3  # At least 3 fingers should be open 

with mp_hands.Hands(min_detection_confidence=0.89, min_tracking_confidence=0.89, max_num_hands=10) as hands:

    #Max Count variable
    max_open_hands = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        hand_count = 0
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Check if the hand is open
                if is_hand_open(hand_landmarks):
                    hand_count += 1

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        #Modification
        if hand_count > max_open_hands:
            max_open_hands = hand_count


        # Display the count 
        cv2.putText(frame, f'Open Hands: {hand_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Max Count: {max_open_hands}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()