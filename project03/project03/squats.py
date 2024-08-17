import cv2
import numpy as np
import threading
from sense_hat import SenseHat

# Initialize Sense HAT
sense = SenseHat()
sense.clear()

# Global variables
squat_count = 0
reset_timer_start = None

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

def reset_counter():
    global squat_count
    squat_count = 0
    print("Counter reset")
    update_sensehat_display()

def update_sensehat_display():
    # Clear the Sense HAT display
    sense.clear()
    
    # Prepare display text
    text = f'Squats: {squat_count}'
    
    # Display text on Sense HAT (fit to 8x8 grid)
    sense.show_message(text, text_colour=[255, 255, 255], back_colour=[0, 0, 0], scroll_speed=0.1)

def start_squats():
    global squat_count, reset_timer_start

    cap = cv2.VideoCapture(0)
    squat_stage = None

    while getattr(threading.currentThread(), "do_run", True):
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.Pose().process(image)
        hand_results = mp_hands.Hands().process(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_height, frame_width, _ = image.shape
        
        # Check for hand gesture to reset counter
        if hand_results.multi_hand_landmarks:
            hand_landmarks_list = []
            for hand_landmarks in hand_results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                hand_landmarks_list.extend(landmarks)

            # Check if all 10 fingers are visible
            if len(hand_landmarks_list) >= 21 * 2:  # 21 landmarks per hand, 2 hands
                if reset_timer_start is None:
                    reset_timer_start = time.time()
                elif time.time() - reset_timer_start >= 5:
                    reset_counter()
                    reset_timer_start = None
            else:
                reset_timer_start = None
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Calculate squat angle
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            angle = calculate_angle(hip, knee, ankle)
            angle = round(angle, 2)
            
            if angle > 160:
                squat_stage = "up"
            if angle < 120 and squat_stage == 'up':
                squat_stage = "down"
                squat_count += 1
                update_sensehat_display()
            
            cv2.putText(image, f'{angle}', 
                        tuple(np.multiply(knee, [frame_width, frame_height]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
        
        # Display exercise type on Sense HAT
        update_sensehat_display()
        
        # Display the result on the screen
        cv2.imshow('Squats', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()