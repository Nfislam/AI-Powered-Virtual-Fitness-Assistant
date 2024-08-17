import cv2
import numpy as np
import time
from PIL import Image
from sense_hat import SenseHat
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

# Initialize SenseHat and camera
sense = SenseHat()
cap = cv2.VideoCapture(0)

# Load the TFLite model with Edge TPU support
interpreter = make_interpreter('movenet_single_pose_lightning_ptq_edgetpu.tflite')
interpreter.allocate_tensors()

# Exercise counters and stages
count_biceps_left = 0
count_biceps_right = 0
stage_biceps_left = None
stage_biceps_right = None
count_squats = 0
stage_squats = None

# Track mode (0 for squats, 1 for bicep curls)
mode = 0
mode_swapped = False  # Flag to prevent indefinite swapping

def draw_text_with_background(image, text, position, font, scale, color, thickness):
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x, y = position
    cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y + 10), (0, 0, 0), -1)
    cv2.putText(image, text, position, font, scale, color, thickness)

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def process_squats(pose, hip_index, knee_index, ankle_index, current_count, current_stage):
    hip = (pose[hip_index][1], pose[hip_index][0])
    knee = (pose[knee_index][1], pose[knee_index][0])
    ankle = (pose[ankle_index][1], pose[ankle_index][0])
    
    angle = calculate_angle(hip, knee, ankle)
    
    if angle > 160:  # Adjusted threshold for 'up' stage
        new_stage = "up"
    elif angle >= 125 and angle <= 105 and current_stage == 'up':  # Half squat
        new_stage = "half_down"
        current_count += 0.5
    elif angle <= 40 and current_stage == 'up':  # Full squat
        new_stage = "down"
        current_count += 1
    else:
        new_stage = current_stage
    
    return current_count, new_stage

def process_curls(pose, shoulder_index, elbow_index, wrist_index, current_count, current_stage):
    shoulder = (pose[shoulder_index][1], pose[shoulder_index][0])
    elbow = (pose[elbow_index][1], pose[elbow_index][0])
    wrist = (pose[wrist_index][1], pose[wrist_index][0])
    
    angle = calculate_angle(shoulder, elbow, wrist)
    
    if angle > 135:
        new_stage = "down"
    elif angle < 30 and current_stage == "down":
        new_stage = "up"
        current_count += 1
    else:
        new_stage = current_stage
    
    return current_count, new_stage

def detect_gesture_switch(pose):
    global mode_swapped
    left_shoulder = pose[5]
    right_shoulder = pose[6]
    left_elbow = pose[7]
    right_elbow = pose[8]

    # Check if both elbows are above their respective shoulders
    if left_elbow[0] < left_shoulder[0] and right_elbow[0] < right_shoulder[0] and not mode_swapped:
        sense.clear()
        mode_swapped = True  # Prevent further swapping
        return True
    elif left_elbow[0] > left_shoulder[0] and right_elbow[0] > right_shoulder[0]:
        mode_swapped = False  # Reset the swap flag when arms are lowered
    return False

def reset_counters():
    global count_biceps_left, count_biceps_right, stage_biceps_left, stage_biceps_right, count_squats, stage_squats
    count_biceps_left = 0
    count_biceps_right = 0
    stage_biceps_left = None
    stage_biceps_right = None
    count_squats = 0
    stage_squats = None

def update_sensehat_display():
    try:
        if mode == 0:
            message = f"S: {count_squats}"
        else:
            message = f"L: {count_biceps_left} R: {count_biceps_right}"
        # Display the message on the Sense HAT
        sense.show_message(message, text_colour=(255, 255, 255), scroll_speed=0.05)
    except Exception as e:
        print(f"Failed to update Sense HAT: {e}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    frame = cv2.flip(frame, 1)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_img = img.resize(common.input_size(interpreter), Image.LANCZOS)
    common.set_input(interpreter, resized_img)

    interpreter.invoke()
    pose = common.output_tensor(interpreter, 0).copy().reshape(-1, 3)

    # Check for the gesture to switch modes
    if detect_gesture_switch(pose):
        mode = 1 if mode == 0 else 0
        reset_counters()
        time.sleep(1)  # Reduced delay to minimize lag

    # Squats processing if mode is 0
    if mode == 0:
        count_squats, stage_squats = process_squats(pose, 11, 13, 15, count_squats, stage_squats)
        draw_text_with_background(frame, f'Squat Count: {count_squats}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Bicep curls processing if mode is 1
    else:
        count_biceps_left, stage_biceps_left = process_curls(pose, 5, 7, 9, count_biceps_left, stage_biceps_left)
        count_biceps_right, stage_biceps_right = process_curls(pose, 6, 8, 10, count_biceps_right, stage_biceps_right)
        draw_text_with_background(frame, f'Left Biceps Count: {count_biceps_left}', (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        draw_text_with_background(frame, f'Right Biceps Count: {count_biceps_right}', (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Continuously update SenseHat display
    update_sensehat_display()

    cv2.imshow('Exercise Tracker', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_counters()

cap.release()
cv2.destroyAllWindows()
