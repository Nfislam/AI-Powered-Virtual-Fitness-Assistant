import cv2
import numpy as np
import threading
from sense_hat import SenseHat
import mediapipe as mp

# Initialize Sense HAT
sense = SenseHat()
sense.clear()

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose()

# Initialize counters
left_curl_count = 0
right_curl_count = 0
left_stage = None
right_stage = None

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ab = a - b
    bc = c - b
    
    cos_ab_bc = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_ab_bc, -1.0, 1.0))
    angle = np.degrees(angle)
    
    return angle

def update_sensehat_display():
    text = f'L: {left_curl_count} R: {right_curl_count}'
    sense.show_message(text, text_colour=[255, 255, 255], back_colour=[0, 0, 0], scroll_speed=0.1)

def start_curls():
    global left_curl_count, right_curl_count, left_stage, right_stage
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Left arm curls
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            left_angle = round(left_angle, 2)
            
            if left_angle > 135:
                left_stage = "down"
            if left_angle < 30 and left_stage == 'down':
                left_stage = "up"
                left_curl_count += 1
                update_sensehat_display()

            cv2.putText(image, f'{left_angle}', 
                        tuple(np.multiply(left_elbow, [frame.shape[1], frame.shape[0]]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Right arm curls
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            right_angle = round(right_angle, 2)
            
            if right_angle > 135:
                right_stage = "down"
            if right_angle < 30 and right_stage == 'down':
                right_stage = "up"
                right_curl_count += 1
                update_sensehat_display()

            cv2.putText(image, f'{right_angle}', 
                        tuple(np.multiply(right_elbow, [frame.shape[1], frame.shape[0]]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display counts on the screen
            cv2.putText(image, f'Left Curls: {left_curl_count}', 
                        (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Right Curls: {right_curl_count}', 
                        (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Curls Tracker', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()