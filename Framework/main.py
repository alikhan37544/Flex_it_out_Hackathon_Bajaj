import cv2
import mediapipe as mp
import numpy as np
import os

from motion_detector import MotionDetector
from thumbs_up import is_thumbs_up
from peace_sign import is_peace_sign
from closed_fist import is_closed_fist
from open_fist import is_open_fist  # New import for open fist detection

# Optional: Force OpenCV to use a different backend if needed.
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Initialize our motion detector.
motion_detector = MotionDetector(velocity_threshold=0.05, acceleration_threshold=0.02, buffer_size=5)

cap = cv2.VideoCapture(10)

def get_hand_region(cx, cy, frame_width, frame_height, region_size=100):
    regions = {
        "Top-Left": (0, 0, region_size, region_size),
        "Top-Right": (frame_width - region_size, 0, region_size, region_size),
        "Bottom-Left": (0, frame_height - region_size, region_size, region_size),
        "Bottom-Right": (frame_width - region_size, frame_height - region_size, region_size, region_size)
    }
    for name, (rx, ry, rw, rh) in regions.items():
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            return name
    return "None"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for a natural feel.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    # Draw corner regions on the frame.
    region_size = 100
    cv2.rectangle(frame, (0, 0), (region_size, region_size), (255, 0, 0), 2)            # Top-left
    cv2.rectangle(frame, (frame_width - region_size, 0), (frame_width, region_size), (0, 255, 0), 2)  # Top-right
    cv2.rectangle(frame, (0, frame_height - region_size), (region_size, frame_height), (0, 0, 255), 2)  # Bottom-left
    cv2.rectangle(frame, (frame_width - region_size, frame_height - region_size), (frame_width, frame_height), (0, 255, 255), 2)  # Bottom-right

    results = hands.process(frame_rgb)
    gesture_text = "None"
    region_text = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Calculate the hand center (average of landmarks) for motion and region detection.
            x_center = np.mean([lm.x for lm in hand_landmarks.landmark])
            y_center = np.mean([lm.y for lm in hand_landmarks.landmark])
            z_val = hand_landmarks.landmark[0].z  # depth coordinate

            # Get the region of the hand in pixel coordinates.
            cx = int(x_center * frame_width)
            cy = int(y_center * frame_height)
            region_text = get_hand_region(cx, cy, frame_width, frame_height, region_size)
            
            position = (x_center, y_center, z_val)
            motion_gesture = motion_detector.update(position)
            
            # Check gestures in priority order.
            if motion_gesture:
                gesture_text = motion_gesture
            elif is_thumbs_up(hand_landmarks):
                gesture_text = "Thumbs Up"
            elif is_peace_sign(hand_landmarks):
                gesture_text = "Peace Sign"
            elif is_closed_fist(hand_landmarks):
                gesture_text = "Closed Fist"
            elif is_open_fist(hand_landmarks):  # Check for open fist only if no other gesture is detected.
                gesture_text = "Open Fist"
            else:
                gesture_text = "Unknown Gesture"

            # Overlay gesture and region info on the frame.
            cv2.putText(frame, f"Gesture: {gesture_text}", (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Region: {region_text}", (10, frame_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit.
        break

cap.release()
cv2.destroyAllWindows()
