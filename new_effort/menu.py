import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os

# ----------------------------
# Pygame Sound Setup (Optional)
# ----------------------------
pygame.mixer.init()

# Background music (looping).
try:
    pygame.mixer.music.load("music.mp3")
    pygame.mixer.music.play(-1)  # Loop indefinitely
except Exception as e:
    print("Error loading background music:", e)

# Selection sound effect.
selection_sound = None
try:
    selection_sound = pygame.mixer.Sound("select.wav")
except Exception as e:
    print("Error loading selection sound:", e)

# ----------------------------
# Menu Items Mapping (2×3 Grid)
# ----------------------------
# This dictionary must match the actual cells you draw:
#   A1, A2, A3, B1, B2, B3
# Any cell can be set to None if you don't have a script for it yet.
menu_items = {
    "A1": "Dance_v3.py",
    "A2": "Dino_Game.py",
    "A3": "Fruit_Ninja.py",
    "B1": "Pushup_Count.py",
    "B2": None,  # placeholder for another script
    "B3": None   # placeholder
}

# ----------------------------
# Threshold & Timer Settings
# ----------------------------
SELECTION_THRESHOLD = 0.5       # Seconds of sustained motion needed to select a cell
MOTION_PIXEL_THRESHOLD = 50     # Minimum motion pixels in that cell to "count" as active
DECAY_RATE = 1.0                # Timer decay rate (seconds per second of no motion)

# We'll store how long each cell has had sufficient motion.
selection_timers = {cell: 0.0 for cell in menu_items.keys()}

# ----------------------------
# Helper Functions
# ----------------------------
def get_hand_region(cx, cy, frame_width, frame_height, grid_cols=2, grid_rows=3):
    """
    Returns the grid cell (e.g., 'A1', 'B2') in which the point (cx, cy) lies.
    For a 2×3 grid:
      - Columns = 2 => A, B
      - Rows = 3    => 1, 2, 3
    """
    cell_width = frame_width // grid_cols
    cell_height = frame_height // grid_rows

    col = cx // cell_width  # 0..(grid_cols-1)
    row = cy // cell_height # 0..(grid_rows-1)

    # Safety clamp if cx/cy is near the boundary
    col = min(col, grid_cols - 1)
    row = min(row, grid_rows - 1)

    col_label = chr(ord('A') + col)  # 'A' for col=0, 'B' for col=1
    row_label = str(row + 1)         # '1' for row=0, '2' for row=1, '3' for row=2
    return col_label + row_label

def overlay_text(frame, text, position, font_scale=0.8, color=(255, 255, 255), thickness=2):
    """Overlays text on the given frame at the specified position."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# ----------------------------
# (Optional) Gesture Detection
# Replace with your own logic or imports
# ----------------------------
def is_thumbs_up(hand_landmarks):
    return False

def is_peace_sign(hand_landmarks):
    return False

def is_closed_fist(hand_landmarks):
    return False

def is_open_fist(hand_landmarks):
    return False

# ----------------------------
# Main Menu Function
# ----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Read first frame for thresholding-based motion detection
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        return
    frame = cv2.resize(frame, (640, 480))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    # We want a 2×3 grid => columns=2, rows=3
    grid_cols = 2
    grid_rows = 3

    last_frame_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width, _ = frame.shape

        # Process hands
        results = hands.process(frame_rgb)

        # Draw the 2×3 grid
        cell_width = frame_width // grid_cols
        cell_height = frame_height // grid_rows
        for r in range(grid_rows):
            for c in range(grid_cols):
                top_left = (c * cell_width, r * cell_height)
                bottom_right = ((c+1) * cell_width, (r+1) * cell_height)
                cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 1)

                cell_label = f"{chr(ord('A') + c)}{r + 1}"  # e.g. 'A1', 'B3'
                cv2.putText(frame, cell_label,
                            (c * cell_width + 5, r * cell_height + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Draw progress bar if partially selected
                if selection_timers[cell_label] > 0:
                    progress = min(selection_timers[cell_label] / SELECTION_THRESHOLD, 1.0)
                    bar_width = int(cell_width * progress)
                    bar_x = c * cell_width
                    bar_y = (r+1) * cell_height - 10
                    cv2.rectangle(frame, (bar_x, bar_y),
                                  (bar_x + bar_width, (r+1)*cell_height),
                                  (0, 255, 0), -1)

        # Thresholding for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        delta = cv2.absdiff(prev_gray, blurred)
        _, thresh_frame = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        prev_gray = blurred.copy()

        # Decay selection timers if not enough motion
        for cell in selection_timers:
            selection_timers[cell] = max(selection_timers[cell] - DECAY_RATE * dt, 0)

        # Process each detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Center of the hand
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                cx = int(np.mean(x_coords) * frame_width)
                cy = int(np.mean(y_coords) * frame_height)

                # Determine which grid cell the hand is over
                current_cell = get_hand_region(cx, cy, frame_width, frame_height,
                                               grid_cols=grid_cols, grid_rows=grid_rows)

                # (Optional) Basic gesture logic
                gesture_text = "Unknown"
                if is_thumbs_up(hand_landmarks):
                    gesture_text = "Thumbs Up"
                elif is_peace_sign(hand_landmarks):
                    gesture_text = "Peace Sign"
                elif is_closed_fist(hand_landmarks):
                    gesture_text = "Closed Fist"
                elif is_open_fist(hand_landmarks):
                    gesture_text = "Open Fist"

                # Overlay gesture & region near the hand
                overlay_text(frame, f"Gesture: {gesture_text}", (cx - 50, cy - 30))
                overlay_text(frame, f"Region: {current_cell}", (cx - 50, cy - 10))

                # Count motion in that cell
                c_idx = ord(current_cell[0]) - ord('A')  # 'A'->0, 'B'->1
                r_idx = int(current_cell[1]) - 1        # '1'->0, '2'->1, '3'->2
                cell_roi = thresh_frame[
                    r_idx * cell_height : (r_idx+1) * cell_height,
                    c_idx * cell_width : (c_idx+1) * cell_width
                ]
                motion_pixels = cv2.countNonZero(cell_roi)

                # If enough motion is detected, accumulate time
                if motion_pixels > MOTION_PIXEL_THRESHOLD:
                    selection_timers[current_cell] += dt

                # If the timer exceeds threshold, launch the script
                if selection_timers[current_cell] >= SELECTION_THRESHOLD:
                    selected_script = menu_items.get(current_cell, None)
                    if selected_script:
                        # Play selection sound if available
                        if selection_sound:
                            selection_sound.play()

                        print(f"Launching {selected_script} from cell {current_cell}!")
                        # Actually run the script
                        os.system(f"python {selected_script}")
                    else:
                        print(f"No script assigned to cell {current_cell}!")

                    # Reset the timer to avoid repeated triggers
                    selection_timers[current_cell] = 0

        # Overlay instructions
        overlay_text(frame, "Hand Gesture Menu System", (10, 30), font_scale=1.0,
                     color=(0, 255, 255), thickness=2)
        overlay_text(frame, "Wave your hand in a cell to select a game!", (10, 60),
                     font_scale=0.6, color=(200, 200, 200), thickness=1)

        cv2.imshow("EyeToy-Inspired Menu", frame)
        # (Optional) show the motion threshold window for debugging:
        # cv2.imshow("Motion Threshold", thresh_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    pygame.quit()

if __name__ == "__main__":
    main()
