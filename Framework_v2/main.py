import cv2
import mediapipe as mp
import numpy as np
import os
import collections
from mediapipe.framework.formats import landmark_pb2

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

hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Initialize our motion detector.
motion_detector = MotionDetector(velocity_threshold=0.05, acceleration_threshold=0.02, buffer_size=5)

cap = cv2.VideoCapture(0)

def get_hand_region(cx, cy, frame_width, frame_height, grid_size=3):
    """
    Returns the region of the hand in a grid format.
    """
    cell_width = frame_width // grid_size
    cell_height = frame_height // grid_size
    col = cx // cell_width
    row = cy // cell_height
    return f"{chr(65 + int(col))}{int(row) + 1}"

def overlay_text(frame, text, position, font_scale=0.8, color=(255, 255, 255), thickness=2):
    """
    Overlays text on the given frame at the specified position.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for a natural feel.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    # Draw grid on the frame.
    grid_size = 3
    cell_width = frame_width // grid_size
    cell_height = frame_height // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            cv2.rectangle(frame, (i * cell_width, j * cell_height), ((i + 1) * cell_width, (j + 1) * cell_height), (255, 255, 255), 1)
            cv2.putText(frame, f"{chr(65 + i)}{j + 1}", (i * cell_width + 5, j * cell_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    results = hands.process(frame_rgb)
    gesture_text_left = "None"
    gesture_text_right = "None"
    region_text_left = "None"
    region_text_right = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Calculate the hand center (average of landmarks) for motion and region detection.
            x_center = np.mean([lm.x for lm in hand_landmarks.landmark])
            y_center = np.mean([lm.y for lm in hand_landmarks.landmark])
            z_val = hand_landmarks.landmark[0].z  # depth coordinate

            # Get the region of the hand in pixel coordinates.
            cx = int(x_center * frame_width)
            cy = int(y_center * frame_height)
            region_text = get_hand_region(cx, cy, frame_width, frame_height, grid_size)
            
            position = (x_center, y_center, z_val)
            motion_gesture = motion_detector.update(position)
            
            # Check gestures in priority order.
            if motion_gesture:
                gesture_text = motion_gesture
            elif is_thumbs_up(hand_landmarks) and not is_closed_fist(hand_landmarks):
                gesture_text = "Thumbs Up"
            elif is_peace_sign(hand_landmarks):
                gesture_text = "Peace Sign"
            elif is_closed_fist(hand_landmarks):
                gesture_text = "Closed Fist"
            elif is_open_fist(hand_landmarks):  # Check for open fist only if no other gesture is detected.
                gesture_text = "Open Fist"
            else:
                gesture_text = "Unknown Gesture"

            # Determine if the hand is left or right.
            if hand_label == 'Left':
                gesture_text_left = gesture_text
                region_text_left = region_text
            else:
                gesture_text_right = gesture_text
                region_text_right = region_text

    # Overlay gesture and region info on the frame.
    overlay_text(frame, f"Left Hand Gesture: {gesture_text_left}", (10, frame_height - 60))
    overlay_text(frame, f"Left Hand Region: {region_text_left}", (10, frame_height - 30))
    overlay_text(frame, f"Right Hand Gesture: {gesture_text_right}", (10, frame_height - 90))
    overlay_text(frame, f"Right Hand Region: {region_text_right}", (10, frame_height - 120))

    cv2.imshow("Hand Gesture Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit.
        break

cap.release()
cv2.destroyAllWindows()