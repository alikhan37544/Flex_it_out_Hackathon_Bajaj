# hand_detection.py

import cv2
import mediapipe as mp
import numpy as np
import os

# If needed, force OpenCV to use the XCB backend (avoids the "wayland" plugin issue)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

def is_closed_fist(hand_landmarks):
    lm = hand_landmarks.landmark
    index_folded = lm[8].y > lm[6].y
    middle_folded = lm[12].y > lm[10].y
    ring_folded = lm[16].y > lm[14].y
    pinky_folded = lm[20].y > lm[18].y
    return index_folded and middle_folded and ring_folded and pinky_folded

def is_open_hand(hand_landmarks):
    lm = hand_landmarks.landmark
    index_open = lm[8].y < lm[6].y
    middle_open = lm[12].y < lm[10].y
    ring_open = lm[16].y < lm[14].y
    pinky_open = lm[20].y < lm[18].y
    return index_open and middle_open and ring_open and pinky_open

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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirror view and convert color space.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    # Draw four corner regions.
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

            # Compute hand center for region detection.
            cx = int(np.mean([lm.x for lm in hand_landmarks.landmark]) * frame_width)
            cy = int(np.mean([lm.y for lm in hand_landmarks.landmark]) * frame_height)
            region_text = get_hand_region(cx, cy, frame_width, frame_height, region_size)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)

            if is_open_hand(hand_landmarks):
                # Get the hand’s center x and wrist z.
                x_center = np.mean([lm.x for lm in hand_landmarks.landmark])
                wrist_z = hand_landmarks.landmark[0].z

                # Check for new gestures.
                if h_wave_detector.detect():
                    gesture_text = "Horizontal Wave"
                    h_wave_detector.reset()  # Reset to avoid continuous triggering.
                elif d_swing_detector.detect():
                    gesture_text = "Depth Swing"
                    d_swing_detector.reset()
                else:
                    gesture_text = "Open Hand"
            elif is_closed_fist(hand_landmarks):
                gesture_text = "Closed Fist"
                h_wave_detector.reset()
                d_swing_detector.reset()
            else:
                gesture_text = "Unknown Gesture"
                h_wave_detector.reset()
                d_swing_detector.reset()

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
